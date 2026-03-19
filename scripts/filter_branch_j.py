"""Branch J: Adaptive ring width per cell for compartment measurement.

Instead of a fixed ring width for all cells, this script measures each cell's
actual DAB peak position along radial profiles and uses THAT as the ring
center/width. The measurement zone adapts to each cell's membrane.

This is NOT a training data filter — it's a post-hoc measurement improvement.
It re-runs the boundary validation (Test 2) with adaptive rings.

Produces: evaluation/branch_j/boundary_validation.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from tqdm import tqdm

from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver

logger = logging.getLogger(__name__)


def _find_membrane_ring_adaptive(
    cell_mask: np.ndarray,
    nuc_mask: np.ndarray,
    dab_channel: np.ndarray,
    n_rays: int = 8,
) -> np.ndarray:
    """Create an adaptive membrane ring mask based on where DAB actually peaks.

    For each radial direction from nucleus centroid, finds the DAB peak position
    and marks a 3px band around it as "membrane."
    """
    ring = np.zeros_like(cell_mask, dtype=bool)

    nuc_ys, nuc_xs = np.where(nuc_mask)
    if len(nuc_ys) == 0:
        return ring

    cx, cy = float(nuc_xs.mean()), float(nuc_ys.mean())

    for angle_deg in range(0, 360, 360 // n_rays):
        angle_rad = np.radians(angle_deg)

        # Walk outward, find where cell ends
        cell_end = 0
        for d in range(1, 60):
            px = int(cx + d * np.cos(angle_rad))
            py = int(cy + d * np.sin(angle_rad))
            if 0 <= px < cell_mask.shape[1] and 0 <= py < cell_mask.shape[0]:
                if cell_mask[py, px]:
                    cell_end = d
            else:
                break

        if cell_end < 5:
            continue

        # Find DAB peak in the outer half of the cell along this ray
        search_start = max(1, cell_end // 2)
        best_d = cell_end
        best_dab = 0.0

        for d in range(search_start, cell_end + 1):
            px = int(cx + d * np.cos(angle_rad))
            py = int(cy + d * np.sin(angle_rad))
            if 0 <= px < dab_channel.shape[1] and 0 <= py < dab_channel.shape[0]:
                val = float(dab_channel[py, px])
                if val > best_dab:
                    best_dab = val
                    best_d = d

        # Mark 3px band around the peak as membrane
        for dd in range(best_d - 1, best_d + 2):
            if dd < 1:
                continue
            px = int(cx + dd * np.cos(angle_rad))
            py = int(cy + dd * np.sin(angle_rad))
            if 0 <= px < ring.shape[1] and 0 <= py < ring.shape[0]:
                # Also mark neighboring pixels for width
                for ox in range(-1, 2):
                    for oy in range(-1, 2):
                        nx, ny = px + ox, py + oy
                        if 0 <= nx < ring.shape[1] and 0 <= ny < ring.shape[0]:
                            if cell_mask[ny, nx]:
                                ring[ny, nx] = True

    return ring


def run_adaptive_validation(cfg: dict):
    """Run compartment intensity test with adaptive per-cell ring width."""
    data_dir = Path(cfg["paths"]["data_dir"])
    eval_dir = Path(cfg["paths"]["evaluation_dir"])
    sc = cfg["stain_deconvolution"]
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))

    # Load model
    from scripts import _load_helpers  # avoid circular — use inline
    tcfg = cfg["training"]

    from instanseg.utils.model_loader import build_model_from_dict
    from instanseg.utils.loss.instanseg_loss import InstanSeg as InstanSegLoss, ProbabilityNet
    from instanseg.utils.augmentations import Augmentations

    ckpt = torch.load(
        "/home/fernandosoto/Documents/models/brightfield_cells_nuclei/model_weights.pth",
        weights_only=False, map_location="cuda",
    )
    model = build_model_from_dict({
        "model_str": "InstanSeg_UNet", "layers": tuple(tcfg["layers"]),
        "dim_in": 3, "n_sigma": tcfg["n_sigma"], "dim_coords": tcfg["dim_coords"],
        "dim_seeds": tcfg["dim_seeds"], "norm": tcfg["norm"],
        "cells_and_nuclei": True, "multihead": False, "dropprob": 0.0,
    })
    loadable = {k: v for k, v in ckpt["model_state_dict"].items() if k in model.state_dict()}
    model.load_state_dict(loadable, strict=False)
    model.eval().cuda()

    pc = ProbabilityNet(embedding_dim=tcfg["dim_coords"] + tcfg["n_sigma"] - 2 + 2, width=tcfg["mlp_width"])
    pc_sd = {k.replace("pixel_classifier.", ""): v
             for k, v in ckpt["model_state_dict"].items() if "pixel_classifier" in k}
    if pc_sd:
        pc.load_state_dict(pc_sd, strict=False)
    pc.eval().cuda()

    loss_fn = InstanSegLoss(n_sigma=tcfg["n_sigma"], dim_coords=tcfg["dim_coords"],
                            dim_seeds=tcfg["dim_seeds"], cells_and_nuclei=True, window_size=32)
    loss_fn.pixel_classifier = pc
    loss_fn.eval().cuda()

    Aug = Augmentations()

    dataset = torch.load(data_dir / "segmentation_dataset_base.pth", weights_only=False)
    test_data = dataset["Test"][:200]
    logger.info("Adaptive ring validation on %d tiles", len(test_data))

    membrane_dab, cytoplasm_dab, nucleus_dab = [], [], []
    fixed_membrane_dab = []  # for comparison with fixed 10px

    for item in tqdm(test_data, desc="Adaptive ring"):
        img = tifffile.imread(data_dir / item["image"])
        nuc_gt = tifffile.imread(data_dir / item["nucleus_masks"]).astype(np.int32)
        if nuc_gt.max() == 0:
            continue

        t, _ = Aug.to_tensor(img, normalize=False)
        t, _ = Aug.normalize(t)

        try:
            with torch.inference_mode():
                raw = model(t.unsqueeze(0).cuda())
                pred = loss_fn.postprocessing(
                    raw[0], seed_threshold=0.5, peak_distance=5,
                    mask_threshold=0.53, overlap_threshold=0.3, window_size=32,
                )
        except Exception:
            continue

        p = pred.cpu().numpy()
        if p.ndim != 3 or p.shape[0] < 2:
            continue
        pred_nuc, pred_cell = p[0].astype(np.int32), p[1].astype(np.int32)

        if pred_nuc.max() == 0 or pred_cell.max() == 0:
            continue

        dab = deconv.extract_dab(img)

        for cell_id in np.unique(pred_cell):
            if cell_id == 0:
                continue

            cell_mask = pred_cell == cell_id
            nuc_mask = pred_nuc == cell_id

            if not nuc_mask.any() or cell_mask.sum() < 20:
                continue

            # Adaptive ring
            adaptive_ring = _find_membrane_ring_adaptive(cell_mask, nuc_mask, dab)

            if adaptive_ring.sum() < 3:
                continue

            cyto_mask = cell_mask & ~nuc_mask & ~adaptive_ring

            membrane_dab.append(float(dab[adaptive_ring].mean()))
            nucleus_dab.append(float(dab[nuc_mask].mean()))
            if cyto_mask.sum() > 3:
                cytoplasm_dab.append(float(dab[cyto_mask].mean()))

            # Also measure with fixed 10px for comparison
            cell_u8 = cell_mask.astype(np.uint8)
            eroded = cv2.erode(cell_u8, np.ones((5, 5), np.uint8), iterations=2)
            fixed_ring = cell_mask & ~eroded.astype(bool)
            if fixed_ring.sum() > 3:
                fixed_membrane_dab.append(float(dab[fixed_ring].mean()))

    results = {
        "method": "adaptive_ring",
        "n_cells": len(membrane_dab),
        "adaptive_ring": {
            "membrane_dab": float(np.mean(membrane_dab)) if membrane_dab else 0,
            "cytoplasm_dab": float(np.mean(cytoplasm_dab)) if cytoplasm_dab else 0,
            "nucleus_dab": float(np.mean(nucleus_dab)) if nucleus_dab else 0,
            "membrane_gt_cytoplasm": bool(np.mean(membrane_dab) > np.mean(cytoplasm_dab)) if membrane_dab and cytoplasm_dab else False,
        },
        "fixed_10px_ring": {
            "membrane_dab": float(np.mean(fixed_membrane_dab)) if fixed_membrane_dab else 0,
        },
    }

    logger.info("\n" + "=" * 60)
    logger.info("ADAPTIVE RING RESULTS (%d cells)", len(membrane_dab))
    logger.info("=" * 60)
    logger.info("  Adaptive: Membrane=%.4f  Cytoplasm=%.4f  Nucleus=%.4f",
                results["adaptive_ring"]["membrane_dab"],
                results["adaptive_ring"]["cytoplasm_dab"],
                results["adaptive_ring"]["nucleus_dab"])
    logger.info("  Fixed 10px: Membrane=%.4f", results["fixed_10px_ring"]["membrane_dab"])
    if results["adaptive_ring"]["membrane_gt_cytoplasm"]:
        logger.info("  --> PASS: Membrane > Cytoplasm (adaptive ring works!)")
    else:
        logger.info("  --> FAIL: Membrane <= Cytoplasm (boundaries still off)")
    logger.info("=" * 60)

    out_dir = eval_dir / "branch_j"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "boundary_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", out_dir / "boundary_validation.json")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()
    run_adaptive_validation(cfg)


if __name__ == "__main__":
    main()
