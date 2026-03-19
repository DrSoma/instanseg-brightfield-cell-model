"""Fixed teacher models v2 — correct APIs for Cellpose v4 and CellSAM.

Runs all three teachers on 500 test tiles, saves masks, builds consensus.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from tqdm import tqdm

from instanseg_brightfield.config import load_config

logger = logging.getLogger(__name__)


def run_instanseg_fluoro(tiles, device="cuda:0"):
    """Low-level API — model.instanseg() not model()."""
    from instanseg import InstanSeg
    from instanseg.utils.augmentations import Augmentations

    model = InstanSeg("fluorescence_nuclei_and_cells", device=device)
    Aug = Augmentations()
    results = {}

    for path, img in tqdm(tiles, desc="InstanSeg fluoro"):
        t, _ = Aug.to_tensor(img, normalize=False)
        t, _ = Aug.normalize(t)
        try:
            with torch.inference_mode():
                out = model.instanseg(t.unsqueeze(0).to(device))
            pred = out[0].cpu().numpy()
            results[str(path)] = pred[1].astype(np.int32) if pred.shape[0] >= 2 else pred[0].astype(np.int32)
        except Exception as e:
            logger.warning("Fluoro failed on %s: %s", path.name, str(e)[:80])
            results[str(path)] = np.zeros(img.shape[:2], dtype=np.int32)

    logger.info("InstanSeg fluoro: %d cells across %d tiles",
                sum(int(v.max()) for v in results.values()), len(results))
    return results


def run_cellpose(tiles, device="cuda:0"):
    """Cellpose v4 API — CellposeModel, not Cellpose."""
    from cellpose.models import CellposeModel

    model = CellposeModel(gpu=(device != "cpu"))
    results = {}

    for path, img in tqdm(tiles, desc="Cellpose"):
        try:
            masks, _, _ = model.eval(img, channels=[0, 0])
            results[str(path)] = masks.astype(np.int32)
        except Exception as e:
            logger.warning("Cellpose failed on %s: %s", path.name, str(e)[:80])
            results[str(path)] = np.zeros(img.shape[:2], dtype=np.int32)

    logger.info("Cellpose: %d cells across %d tiles",
                sum(int(v.max()) for v in results.values()), len(results))
    return results


def run_cellsam(tiles, device="cuda:1"):
    """CellSAM — needs get_model() first, then pass model to segment_cellular_image()."""
    from cellSAM import get_model, segment_cellular_image

    logger.info("Loading CellSAM model...")
    model = get_model()
    model = model.to(device)
    model.eval()
    results = {}

    for path, img in tqdm(tiles, desc="CellSAM"):
        try:
            mask = segment_cellular_image(img, model=model, device=device)
            if isinstance(mask, tuple):
                mask = mask[0]
            results[str(path)] = np.asarray(mask, dtype=np.int32)
        except Exception as e:
            logger.warning("CellSAM failed on %s: %s", path.name, str(e)[:80])
            results[str(path)] = np.zeros(img.shape[:2], dtype=np.int32)

    logger.info("CellSAM: %d cells across %d tiles",
                sum(int(v.max()) for v in results.values()), len(results))
    return results


def build_consensus(all_preds, min_agreement=2):
    """Where >= min_agreement teachers agree on cell presence."""
    all_tiles = set()
    for preds in all_preds.values():
        all_tiles.update(preds.keys())

    consensus = {}
    for tp in tqdm(sorted(all_tiles), desc="Consensus"):
        masks = [preds[tp] for preds in all_preds.values() if tp in preds and preds[tp].max() > 0]
        if len(masks) < min_agreement:
            h, w = masks[0].shape if masks else (512, 512)
            consensus[tp] = np.zeros((h, w), dtype=np.int32)
            continue

        # Binary agreement
        binary = [(m > 0).astype(np.float32) for m in masks]
        agreement = np.sum(binary, axis=0)
        agreed = agreement >= min_agreement

        # Use mask with most cells as instance labels
        best = max(masks, key=lambda m: m.max())
        consensus[tp] = np.where(agreed, best, 0).astype(np.int32)

    logger.info("Consensus: %d cells across %d tiles",
                sum(int(v.max()) for v in consensus.values()), len(consensus))
    return consensus


def save_masks(results, out_dir, data_dir):
    """Save prediction masks as TIFFs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for tp_str, mask in results.items():
        tp = Path(tp_str)
        # Extract slide name from path: data/tiles/SLIDE/tile.png → SLIDE
        parts = tp.relative_to(data_dir / "tiles").parts if (data_dir / "tiles") in tp.parents else tp.parts
        slide = parts[0] if len(parts) >= 2 else "unknown"
        stem = tp.stem

        slide_dir = out_dir / slide
        slide_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(slide_dir / f"{stem}_cells.tiff"), mask.astype(np.uint16), compression="zlib")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])

    # Load tiles
    dataset = torch.load(data_dir / "segmentation_dataset_base.pth", weights_only=False)
    tile_items = dataset["Test"][:500]

    tiles = []
    for item in tile_items:
        img_path = data_dir / item["image"]
        if img_path.exists():
            tiles.append((img_path, tifffile.imread(img_path)))

    logger.info("Loaded %d tiles", len(tiles))
    all_preds = {}

    # 1. InstanSeg fluoro (GPU 0)
    logger.info("=== InstanSeg fluorescence_nuclei_and_cells ===")
    try:
        fluoro = run_instanseg_fluoro(tiles, device="cuda:0")
        if sum(int(v.max()) for v in fluoro.values()) > 0:
            all_preds["instanseg_fluoro"] = fluoro
            save_masks(fluoro, data_dir / "teacher_predictions" / "instanseg_fluoro", data_dir)
    except Exception as e:
        logger.error("InstanSeg fluoro: %s", e)
    torch.cuda.empty_cache()

    # 2. Cellpose (GPU 0)
    logger.info("=== Cellpose cyto3 ===")
    try:
        cellpose = run_cellpose(tiles, device="cuda:0")
        if sum(int(v.max()) for v in cellpose.values()) > 0:
            all_preds["cellpose"] = cellpose
            save_masks(cellpose, data_dir / "teacher_predictions" / "cellpose", data_dir)
    except Exception as e:
        logger.error("Cellpose: %s", e)
    torch.cuda.empty_cache()

    # 3. CellSAM (GPU 1)
    logger.info("=== CellSAM ===")
    try:
        cellsam = run_cellsam(tiles, device="cuda:1")
        if sum(int(v.max()) for v in cellsam.values()) > 0:
            all_preds["cellsam"] = cellsam
            save_masks(cellsam, data_dir / "teacher_predictions" / "cellsam", data_dir)
    except Exception as e:
        logger.error("CellSAM: %s", e)
    torch.cuda.empty_cache()

    logger.info("Teachers with results: %s", list(all_preds.keys()))

    # Consensus
    if len(all_preds) >= 2:
        logger.info("Building consensus from %d teachers", len(all_preds))
        consensus = build_consensus(all_preds, min_agreement=2)
        save_masks(consensus, data_dir / "teacher_predictions" / "consensus", data_dir)

    # Summary
    summary = {
        "teachers": list(all_preds.keys()),
        "tiles": len(tiles),
        "cells_per_teacher": {k: sum(int(v.max()) for v in vs.values()) for k, vs in all_preds.items()},
    }
    Path("evaluation").mkdir(exist_ok=True)
    with open("evaluation/teacher_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
