#!/usr/bin/env python3
"""Layer 2: GPU conditional dilation — expand cell boundaries toward DAB membrane.

Implements DAB-guided boundary refinement that shifts cell mask edges by up to
``MAX_EXPAND_PX`` pixels into DAB-positive regions.  Uses bar-filter response
as the expansion criterion for orientation-aware membrane snapping.

Two refinement methods compared:
  A) Conditional dilation with centroid-distance Voronoi tie-breaking
  B) DAB-energy watershed re-segmentation (CSGO-style)

Validates each by computing the fixed-ring membrane > cytoplasm gap and
compares against the current unrefined gap (-0.020) and the bar-filter
baseline (+0.127).

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python scripts/15_boundary_refinement.py [--n-tiles 200]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi
import tifffile
import torch
import torch.nn.functional as F
from skimage.segmentation import watershed
from tqdm import tqdm

from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Parameters ────────────────────────────────────────────────────────────
MAX_EXPAND_PX = 4           # Maximum boundary expansion (px at read level)
DAB_THRESHOLD = 0.08        # Minimum DAB OD for expansion
BAR_THRESHOLD = 0.0         # Minimum bar-filter response (0 = any positive response)
RING_ERODE_KSIZE = 5        # Kernel size for fixed-ring measurement
RING_ERODE_ITER = 2         # Erosion iterations (~10px ring)


# ═══════════════════════════════════════════════════════════════════════════
# BAR-FILTER (reused from comprehensive_membrane_validation.py)
# ═══════════════════════════════════════════════════════════════════════════

def build_bar_kernels(
    n_orientations: int = 8,
    ksize: int = 25,
    sigma_long: float = 5.0,
    sigma_short: float = 1.0,
) -> torch.Tensor:
    """Build oriented Gaussian bar-filter kernels for membrane detection."""
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    kernels = []
    for i in range(n_orientations):
        theta = i * np.pi / n_orientations
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t
        g = np.exp(-0.5 * (x_rot**2 / sigma_long**2 + y_rot**2 / sigma_short**2))
        g -= g.mean()
        g /= np.abs(g).sum() + 1e-10
        kernels.append(g)
    kernels_np = np.stack(kernels)[:, np.newaxis, :, :]
    return torch.from_numpy(kernels_np).float()


def apply_bar_filters_gpu(
    dab_batch: torch.Tensor,
    kernels: torch.Tensor,
) -> torch.Tensor:
    """Apply all bar-filter orientations via single grouped conv2d.

    Args:
        dab_batch: (B, 1, H, W) DAB on GPU.
        kernels: (N, 1, ksize, ksize) on GPU.

    Returns:
        (B, H, W) max response across orientations.
    """
    pad = kernels.shape[-1] // 2
    responses = F.conv2d(dab_batch, kernels, padding=pad)
    return responses.max(dim=1).values


# ═══════════════════════════════════════════════════════════════════════════
# METHOD A: CONDITIONAL EROSION — SHRINK TO MEMBRANE PEAK
# ═══════════════════════════════════════════════════════════════════════════

def refine_conditional_erosion(
    cell_labels: np.ndarray,
    nuc_labels: np.ndarray,
    dab: np.ndarray,
    bar_response: np.ndarray,
    max_shrink_px: int = MAX_EXPAND_PX,
    dab_threshold: float = DAB_THRESHOLD,
    bar_threshold: float = BAR_THRESHOLD,
    device: str = "cuda",
) -> np.ndarray:
    """Shrink cell boundaries INWARD by peeling off the outer non-membrane shell.

    The cell boundaries are ~4px PAST the membrane peak (too big). This method
    removes boundary pixels where DAB is low (extracellular fringe), stopping
    at the membrane where DAB is high.

    Uses min_pool2d (= erosion on labels) with DAB-guided stopping.
    """
    H, W = cell_labels.shape
    refined = cell_labels.copy()

    for iteration in range(max_shrink_px):
        # Find current boundary pixels (cell pixels adjacent to background or other cells)
        labels_t = torch.from_numpy(refined.astype(np.float32)).to(device)
        labels_4d = labels_t.unsqueeze(0).unsqueeze(0)

        # Erode: min_pool2d with background (0) treated as -inf won't work directly.
        # Instead: for each cell pixel, check if ANY neighbor is background or different label.
        # A pixel is a boundary pixel if its min neighbor != its own label.
        min_neighbor = -F.max_pool2d(-labels_4d, kernel_size=3, stride=1, padding=1)
        min_neighbor = min_neighbor.squeeze().cpu().numpy().astype(np.int32)

        # Boundary = pixels where min_neighbor != self (i.e., adjacent to bg or other cell)
        is_boundary = (refined > 0) & (min_neighbor != refined)

        # Only erode where DAB is LOW and bar-filter response is LOW
        # (i.e., this boundary pixel is NOT on the membrane — it's the overshoot fringe)
        low_dab = dab < dab_threshold * (1 + 0.5 * iteration)  # progressive: relax threshold in later iterations
        low_bar = bar_response <= bar_threshold

        # Pixels to remove: boundary AND (low DAB OR low bar-filter)
        should_erode = is_boundary & (low_dab | low_bar)

        # Safety: never erode into the nucleus
        is_nucleus = nuc_labels > 0
        should_erode = should_erode & ~is_nucleus

        # Safety: never make a cell smaller than its nucleus
        for cid in np.unique(refined[should_erode]):
            if cid == 0:
                continue
            cell_mask = refined == cid
            nuc_mask = nuc_labels == cid
            erode_mask = should_erode & (refined == cid)
            remaining = cell_mask.sum() - erode_mask.sum()
            if nuc_mask.any() and remaining < nuc_mask.sum() * 1.2:
                # Would make cell smaller than 1.2x nucleus — skip this cell
                should_erode[erode_mask] = False

        if not should_erode.any():
            break

        # Remove boundary pixels
        refined[should_erode] = 0

    return refined


# ═══════════════════════════════════════════════════════════════════════════
# METHOD B: DAB-ENERGY WATERSHED (CSGO-STYLE)
# ═══════════════════════════════════════════════════════════════════════════

def refine_dab_watershed(
    cell_labels: np.ndarray,
    nuc_labels: np.ndarray,
    dab: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Re-run watershed using DAB as energy landscape.

    Seeds are the existing cell labels (minus boundaries). The watershed
    grows cells using DAB intensity as the energy, so boundaries naturally
    snap to DAB membrane ridges.
    """
    from skimage.filters import gaussian

    # Smooth DAB for watershed energy
    dab_smooth = gaussian(dab, sigma=sigma)

    # Energy: invert DAB so high DAB = low energy = boundary
    # Watershed finds ridges (high energy boundaries) between basins
    energy = 1.0 - dab_smooth / (dab_smooth.max() + 1e-8)

    # Seeds: erode existing labels to remove current boundaries
    kernel = np.ones((3, 3), np.uint8)
    markers = cv2.erode(cell_labels.astype(np.uint16), kernel, iterations=2)

    # Constrain to region where cells exist (don't expand into stroma)
    mask = ndi.binary_dilation(cell_labels > 0, iterations=MAX_EXPAND_PX + 1)

    # Watershed with DAB energy
    refined = watershed(
        energy.astype(np.float64),
        markers=markers.astype(np.int32),
        mask=mask,
        compactness=0.0,
    )

    # Enforce nucleus containment
    for nid in np.unique(nuc_labels):
        if nid == 0:
            continue
        refined[nuc_labels == nid] = nid

    return refined.astype(np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# MEASUREMENT: FIXED-RING GAP
# ═══════════════════════════════════════════════════════════════════════════

def measure_fixed_ring_gap(
    dab: np.ndarray,
    cell_labels: np.ndarray,
    nuc_labels: np.ndarray,
    min_cell_area: int = 20,
) -> dict:
    """Measure membrane vs cytoplasm DAB using fixed 10px morphological ring.

    This is the TEST that currently FAILS (gap = -0.020).
    """
    membrane_dabs = []
    cytoplasm_dabs = []
    nucleus_dabs = []

    for cell_id in np.unique(cell_labels):
        if cell_id == 0:
            continue
        cell_mask = cell_labels == cell_id
        nuc_mask = nuc_labels == cell_id

        if cell_mask.sum() < min_cell_area or not nuc_mask.any():
            continue

        # Membrane ring: erode cell mask, ring = cell - eroded
        cell_u8 = cell_mask.astype(np.uint8)
        eroded = cv2.erode(cell_u8, np.ones((RING_ERODE_KSIZE, RING_ERODE_KSIZE), np.uint8),
                           iterations=RING_ERODE_ITER)
        ring = cell_mask & ~eroded.astype(bool)

        # Cytoplasm: interior minus nucleus minus ring
        cytoplasm = cell_mask & eroded.astype(bool) & ~nuc_mask

        if ring.sum() < 3 or cytoplasm.sum() < 3:
            continue

        membrane_dabs.append(float(dab[ring].mean()))
        cytoplasm_dabs.append(float(dab[cytoplasm].mean()))
        if nuc_mask.sum() > 3:
            nucleus_dabs.append(float(dab[nuc_mask].mean()))

    if not membrane_dabs:
        return {"membrane_dab_mean": 0, "cytoplasm_dab_mean": 0, "gap": 0, "n_cells": 0}

    mem_mean = float(np.mean(membrane_dabs))
    cyto_mean = float(np.mean(cytoplasm_dabs))
    return {
        "membrane_dab_mean": round(mem_mean, 4),
        "cytoplasm_dab_mean": round(cyto_mean, 4),
        "gap": round(mem_mean - cyto_mean, 4),
        "n_cells": len(membrane_dabs),
        "status": "PASS" if mem_mean > cyto_mean else "FAIL",
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING (same as comprehensive_membrane_validation)
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_postprocess(cfg: dict, device: str):
    """Load trained InstanSeg model + pixel classifier + loss module."""
    from instanseg.utils.model_loader import build_model_from_dict
    from instanseg.utils.loss.instanseg_loss import InstanSeg as InstanSegLoss, ProbabilityNet
    from instanseg.utils.augmentations import Augmentations

    tcfg = cfg["training"]
    model_dir = Path(cfg["paths"]["model_dir"])

    # Try multiple checkpoint paths
    ckpt_paths = [
        model_dir / "brightfield_cells_nuclei_v1_finetuned" / "model_weights.pth",
        Path("/home/fernandosoto/Documents/models/brightfield_cells_nuclei/model_weights.pth"),
    ]
    ckpt_path = next((p for p in ckpt_paths if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found at: {ckpt_paths}")

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model = build_model_from_dict({
        "model_str": "InstanSeg_UNet", "layers": tuple(tcfg["layers"]),
        "dim_in": 3, "n_sigma": tcfg["n_sigma"], "dim_coords": tcfg["dim_coords"],
        "dim_seeds": tcfg["dim_seeds"], "norm": tcfg["norm"],
        "cells_and_nuclei": True, "multihead": False, "dropprob": 0.0,
    })
    loadable = {k: v for k, v in ckpt["model_state_dict"].items() if k in model.state_dict()}
    model.load_state_dict(loadable, strict=False)
    model.eval().to(device)

    pc = ProbabilityNet(
        embedding_dim=tcfg["dim_coords"] + tcfg["n_sigma"] - 2 + 2,
        width=tcfg["mlp_width"],
    )
    pc_sd = {k.replace("pixel_classifier.", ""): v
             for k, v in ckpt["model_state_dict"].items() if "pixel_classifier" in k}
    if pc_sd:
        pc.load_state_dict(pc_sd, strict=False)
    pc.eval().to(device)

    loss_fn = InstanSegLoss(
        n_sigma=tcfg["n_sigma"], dim_coords=tcfg["dim_coords"],
        dim_seeds=tcfg["dim_seeds"], cells_and_nuclei=True, window_size=32,
    )
    loss_fn.pixel_classifier = pc
    loss_fn.eval().to(device)

    Aug = Augmentations()
    return model, loss_fn, Aug


def predict_tile(model, loss_fn, Aug, img: np.ndarray, device: str):
    """Run inference on one tile, return (pred_nuc, pred_cell) int32 masks."""
    t, _ = Aug.to_tensor(img, normalize=False)
    t, _ = Aug.normalize(t)

    with torch.inference_mode():
        raw = model(t.unsqueeze(0).to(device))
        pred = loss_fn.postprocessing(
            raw[0], seed_threshold=0.5, peak_distance=5,
            mask_threshold=0.53, overlap_threshold=0.3, window_size=32,
        )

    p = pred.cpu().numpy()
    if p.ndim == 3 and p.shape[0] >= 2:
        return p[0].astype(np.int32), p[1].astype(np.int32)
    return p[0].astype(np.int32), np.zeros_like(p[0], dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Layer 2: Boundary refinement")
    parser.add_argument("--n-tiles", type=int, default=200)
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    t0 = time.time()
    device = args.device
    n_tiles = args.n_tiles

    logger.info("=" * 72)
    logger.info("LAYER 2: BOUNDARY REFINEMENT — Conditional Dilation + DAB Watershed")
    logger.info("Max expansion: %d px | DAB threshold: %.2f | Device: %s",
                MAX_EXPAND_PX, DAB_THRESHOLD, device)
    logger.info("=" * 72)

    # Setup
    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    eval_dir = Path(cfg["paths"]["evaluation_dir"])
    sc = cfg["stain_deconvolution"]
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))

    # Load model
    logger.info("Loading InstanSeg model...")
    model, loss_fn, Aug = load_model_and_postprocess(cfg, device)
    logger.info("Model loaded (%.1fs)", time.time() - t0)

    # Bar-filter kernels on GPU
    bar_kernels = build_bar_kernels(n_orientations=8, ksize=25,
                                     sigma_long=5.0, sigma_short=1.0).to(device)

    # Load test data
    dataset_path = data_dir / "segmentation_dataset_base.pth"
    if not dataset_path.exists():
        dataset_path = data_dir / "segmentation_dataset.pth"
    dataset = torch.load(dataset_path, weights_only=False)
    test_data = dataset["Test"][:n_tiles]
    logger.info("Test tiles: %d", len(test_data))

    # Accumulators for each method
    results = {
        "original": {"membrane": [], "cytoplasm": [], "n_cells": 0},
        "method_a_erosion": {"membrane": [], "cytoplasm": [], "n_cells": 0},
        "method_b_watershed": {"membrane": [], "cytoplasm": [], "n_cells": 0},
    }

    tiles_processed = 0
    skipped = 0

    for item in tqdm(test_data, desc="Refining boundaries"):
        img = tifffile.imread(data_dir / item["image"])
        if img is None or img.size == 0:
            skipped += 1
            continue

        # Predict
        try:
            pred_nuc, pred_cell = predict_tile(model, loss_fn, Aug, img, device)
        except Exception as e:
            logger.debug("Prediction failed: %s", e)
            skipped += 1
            continue

        if pred_cell.max() == 0 or pred_nuc.max() == 0:
            skipped += 1
            continue

        # DAB extraction
        dab = deconv.extract_dab(img)

        # Bar-filter on GPU
        dab_t = torch.from_numpy(dab).float().to(device).unsqueeze(0).unsqueeze(0)
        bar_resp_t = apply_bar_filters_gpu(dab_t, bar_kernels)
        bar_resp = bar_resp_t.squeeze().cpu().numpy()

        # ── Measure ORIGINAL (unrefined) ──
        orig_result = measure_fixed_ring_gap(dab, pred_cell, pred_nuc)
        if orig_result["n_cells"] > 0:
            results["original"]["membrane"].extend([orig_result["membrane_dab_mean"]] * orig_result["n_cells"])
            results["original"]["cytoplasm"].extend([orig_result["cytoplasm_dab_mean"]] * orig_result["n_cells"])
            results["original"]["n_cells"] += orig_result["n_cells"]

        # ── Method A: Conditional Erosion (shrink to membrane) ──
        refined_a = refine_conditional_erosion(
            pred_cell, pred_nuc, dab, bar_resp,
            max_shrink_px=MAX_EXPAND_PX,
            dab_threshold=DAB_THRESHOLD,
            bar_threshold=BAR_THRESHOLD,
            device=device,
        )
        result_a = measure_fixed_ring_gap(dab, refined_a, pred_nuc)
        if result_a["n_cells"] > 0:
            results["method_a_erosion"]["membrane"].extend([result_a["membrane_dab_mean"]] * result_a["n_cells"])
            results["method_a_erosion"]["cytoplasm"].extend([result_a["cytoplasm_dab_mean"]] * result_a["n_cells"])
            results["method_a_erosion"]["n_cells"] += result_a["n_cells"]

        # ── Method B: DAB Watershed ──
        refined_b = refine_dab_watershed(pred_cell, pred_nuc, dab, sigma=1.0)
        result_b = measure_fixed_ring_gap(dab, refined_b, pred_nuc)
        if result_b["n_cells"] > 0:
            results["method_b_watershed"]["membrane"].extend([result_b["membrane_dab_mean"]] * result_b["n_cells"])
            results["method_b_watershed"]["cytoplasm"].extend([result_b["cytoplasm_dab_mean"]] * result_b["n_cells"])
            results["method_b_watershed"]["n_cells"] += result_b["n_cells"]

        tiles_processed += 1

    # ── Summarize ──
    logger.info("=" * 72)
    logger.info("RESULTS: %d tiles processed, %d skipped", tiles_processed, skipped)
    logger.info("=" * 72)

    summary = {
        "n_tiles": tiles_processed,
        "n_skipped": skipped,
        "max_expand_px": MAX_EXPAND_PX,
        "dab_threshold": DAB_THRESHOLD,
        "bar_threshold": BAR_THRESHOLD,
        "methods": {},
    }

    for method_name, data in results.items():
        if data["membrane"]:
            mem_mean = float(np.mean(data["membrane"]))
            cyto_mean = float(np.mean(data["cytoplasm"]))
            gap = mem_mean - cyto_mean
            status = "PASS" if gap > 0 else "FAIL"
        else:
            mem_mean = cyto_mean = gap = 0.0
            status = "NO DATA"

        summary["methods"][method_name] = {
            "membrane_dab_mean": round(mem_mean, 4),
            "cytoplasm_dab_mean": round(cyto_mean, 4),
            "gap": round(gap, 4),
            "status": status,
            "n_cells": data["n_cells"],
        }

        logger.info(
            "%-25s | membrane=%.4f  cytoplasm=%.4f  gap=%+.4f  %s  (%d cells)",
            method_name, mem_mean, cyto_mean, gap, status, data["n_cells"],
        )

    # Compare against bar-filter baseline
    logger.info("-" * 72)
    logger.info("REFERENCE: bar-filter baseline gap = +0.1269 (from boundary_validation.json)")
    logger.info("REFERENCE: original fixed-ring gap = -0.0201 (from boundary_validation.json)")
    logger.info("-" * 72)

    best_method = max(summary["methods"].items(), key=lambda x: x[1]["gap"])
    logger.info("BEST METHOD: %s (gap = %+.4f)", best_method[0], best_method[1]["gap"])

    if best_method[1]["gap"] > 0.05:
        logger.info("GO: Gap > +0.05 — boundary refinement is working!")
    elif best_method[1]["gap"] > 0:
        logger.info("MARGINAL: Gap positive but < +0.05 — may need parameter tuning")
    else:
        logger.info("NO-GO: Gap still negative — boundary refinement insufficient")

    # Save results
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / "boundary_refinement_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to %s", output_path)

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
