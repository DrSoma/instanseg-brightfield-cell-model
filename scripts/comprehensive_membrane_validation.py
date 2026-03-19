#!/usr/bin/env python3
"""Comprehensive membrane measurement validation — all 3 tasks in one pass.

Combines three independent approaches to validate cell boundaries:
  1. Aperio bar-filter membrane detection (FDA-cleared approach for HER2)
  2. Adaptive per-cell ring width (radial DAB peak finding)
  3. Clinical H-score concordance (model boundaries vs nucleus expansion)

All approaches share a single model load and inference pass per tile.
Uses GPU 1 by default (GPU 0 reserved for SegFormer training).

Usage:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=1 \
    python scripts/comprehensive_membrane_validation.py [--n-tiles 200]

Note: CUDA_VISIBLE_DEVICES=1 remaps physical GPU 1 to cuda:0 inside this
process.  InstanSeg postprocessing creates tensors on the default CUDA device,
so this avoids a cuda:0 / cuda:1 mismatch.
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
from tqdm import tqdm

from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. APERIO BAR-FILTER MEMBRANE DETECTION
# ═══════════════════════════════════════════════════════════════════════════


def build_bar_kernels(
    n_orientations: int = 8,
    ksize: int = 25,
    sigma_long: float = 5.0,
    sigma_short: float = 1.0,
) -> torch.Tensor:
    """Build oriented Gaussian bar-filter kernels for membrane detection.

    Creates elongated Gaussian kernels rotated at evenly-spaced angles.
    At 0.5 μm/px, sigma_long=5 (2.5 μm) and sigma_short=1 (0.5 μm) match
    the measured membrane width (~5 μm FWHM).

    Returns:
        Tensor of shape (n_orientations, 1, ksize, ksize) ready for conv2d.
    """
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    kernels = []

    for i in range(n_orientations):
        theta = i * np.pi / n_orientations  # 0 to π (180° covers all due to symmetry)

        # Rotate coordinates
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = x * cos_t + y * sin_t      # along bar direction
        y_rot = -x * sin_t + y * cos_t     # perpendicular to bar

        # Anisotropic Gaussian
        g = np.exp(-0.5 * (x_rot ** 2 / sigma_long ** 2 + y_rot ** 2 / sigma_short ** 2))

        # Zero-mean: subtract mean so filter responds to edges, not flat regions
        g -= g.mean()
        # Normalize L1
        g /= np.abs(g).sum() + 1e-10

        kernels.append(g)

    kernels_np = np.stack(kernels)[:, np.newaxis, :, :]  # (N, 1, H, W)
    return torch.from_numpy(kernels_np).float()


def apply_bar_filters_gpu(
    dab_batch: torch.Tensor,
    kernels: torch.Tensor,
) -> torch.Tensor:
    """Apply all bar-filter orientations in one conv2d call.

    Args:
        dab_batch: (B, 1, H, W) DAB channel tensor on GPU.
        kernels: (N_orientations, 1, ksize, ksize) on GPU.

    Returns:
        (B, H, W) max response across orientations per pixel.
    """
    pad = kernels.shape[-1] // 2
    # F.conv2d with groups=1: each kernel applied to the single input channel
    # Output: (B, N_orientations, H, W)
    responses = F.conv2d(dab_batch, kernels, padding=pad)
    # Max across orientations
    max_response, _ = responses.max(dim=1)  # (B, H, W)
    return max_response


def measure_bar_filter_per_cell(
    dab: np.ndarray,
    bar_response: np.ndarray,
    cell_labels: np.ndarray,
    nuc_labels: np.ndarray,
    min_cell_area: int = 20,
) -> dict:
    """Measure membrane DAB using bar-filter response as weights per cell.

    For each cell, computes:
      - weighted_membrane_dab: mean DAB weighted by bar-filter response
      - peak_membrane_dab: mean DAB at pixels where bar-response > median
      - cytoplasm_dab: mean DAB in non-membrane interior
      - completeness: fraction of boundary with strong bar response
    """
    membrane_dab_list = []
    cytoplasm_dab_list = []
    nucleus_dab_list = []
    completeness_list = []

    # Threshold for "membrane" pixels: positive bar response
    bar_positive = bar_response > 0

    for cell_id in np.unique(cell_labels):
        if cell_id == 0:
            continue

        cell_mask = cell_labels == cell_id
        nuc_mask = nuc_labels == cell_id

        if cell_mask.sum() < min_cell_area or not nuc_mask.any():
            continue

        # Membrane zone: cell pixels with positive bar-filter response
        cell_membrane = cell_mask & bar_positive
        # Cytoplasm: cell minus nucleus minus membrane
        cell_cyto = cell_mask & ~nuc_mask & ~cell_membrane

        if cell_membrane.sum() < 3:
            continue

        # Weighted membrane DAB (bar response as weights, clamped ≥ 0)
        weights = np.clip(bar_response[cell_mask], 0, None)
        dab_vals = dab[cell_mask]
        if weights.sum() > 0:
            weighted_dab = float(np.average(dab_vals, weights=weights))
        else:
            weighted_dab = float(dab_vals.mean())

        membrane_dab_list.append(weighted_dab)
        nucleus_dab_list.append(float(dab[nuc_mask].mean()))

        if cell_cyto.sum() > 3:
            cytoplasm_dab_list.append(float(dab[cell_cyto].mean()))

        # Completeness: fraction of cell boundary with bar response
        boundary = cell_mask.astype(np.uint8)
        eroded = cv2.erode(boundary, np.ones((3, 3), np.uint8), iterations=1)
        edge = (boundary - eroded).astype(bool)
        if edge.sum() > 0:
            completeness_list.append(float(bar_positive[edge].mean()))

    return {
        "membrane_dab": membrane_dab_list,
        "cytoplasm_dab": cytoplasm_dab_list,
        "nucleus_dab": nucleus_dab_list,
        "completeness": completeness_list,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. ADAPTIVE PER-CELL RING WIDTH
# ═══════════════════════════════════════════════════════════════════════════


def measure_adaptive_ring_per_cell(
    dab: np.ndarray,
    cell_labels: np.ndarray,
    nuc_labels: np.ndarray,
    n_rays: int = 16,
    band_width: int = 3,
    min_cell_area: int = 20,
) -> dict:
    """Vectorized adaptive ring measurement using radial DAB peak finding.

    For each cell, casts n_rays radial rays from nucleus centroid outward,
    finds where DAB peaks along each ray in the outer half, and measures
    DAB in a band around the peak.

    Uses scipy.ndimage.map_coordinates for fast sub-pixel sampling.
    """
    membrane_dab_list = []
    cytoplasm_dab_list = []
    nucleus_dab_list = []
    peak_distances = []

    h, w = dab.shape
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    for cell_id in np.unique(cell_labels):
        if cell_id == 0:
            continue

        cell_mask = cell_labels == cell_id
        nuc_mask = nuc_labels == cell_id

        if cell_mask.sum() < min_cell_area or not nuc_mask.any():
            continue

        # Centroid of nucleus
        nuc_ys, nuc_xs = np.where(nuc_mask)
        cy, cx = float(nuc_ys.mean()), float(nuc_xs.mean())

        # Find cell extent along each ray
        max_dist = 60
        distances = np.arange(1, max_dist + 1, dtype=np.float64)

        # Pre-compute all ray coordinates: (n_rays, max_dist) for x and y
        ray_xs = cx + np.outer(cos_a, distances)  # (n_rays, max_dist)
        ray_ys = cy + np.outer(sin_a, distances)

        # Clip to image bounds
        ray_xs_clip = np.clip(ray_xs, 0, w - 1)
        ray_ys_clip = np.clip(ray_ys, 0, h - 1)

        # Sample cell_mask along rays (integer coords for speed)
        ray_xi = ray_xs_clip.astype(int)
        ray_yi = ray_ys_clip.astype(int)
        in_cell = cell_mask[ray_yi, ray_xi]  # (n_rays, max_dist) bool

        # Find cell_end per ray (last distance still in cell)
        cell_ends = np.zeros(n_rays, dtype=int)
        for r in range(n_rays):
            valid = np.where(in_cell[r])[0]
            if len(valid) > 0:
                cell_ends[r] = int(valid[-1]) + 1  # convert to distance

        # Sample DAB along rays
        dab_along_rays = dab[ray_yi, ray_xi]  # (n_rays, max_dist)

        # Find DAB peak in outer half of each ray
        adaptive_ring = np.zeros_like(cell_mask, dtype=bool)
        ray_peak_dabs = []

        for r in range(n_rays):
            ce = cell_ends[r]
            if ce < 5:
                continue

            search_start = max(0, ce // 2)
            ray_dab = dab_along_rays[r, search_start:ce]
            if len(ray_dab) == 0:
                continue

            peak_idx = search_start + int(np.argmax(ray_dab))
            peak_distances.append(peak_idx + 1)

            # Mark band around peak
            for dd in range(max(0, peak_idx - band_width), min(max_dist, peak_idx + band_width + 1)):
                px = ray_xi[r, dd]
                py = ray_yi[r, dd]
                if cell_mask[py, px]:
                    adaptive_ring[py, px] = True
                    # Also mark 1px neighbors for width
                    for ox in range(-1, 2):
                        for oy in range(-1, 2):
                            nx, ny = px + ox, py + oy
                            if 0 <= nx < w and 0 <= ny < h and cell_mask[ny, nx]:
                                adaptive_ring[ny, nx] = True

        if adaptive_ring.sum() < 3:
            continue

        cyto_mask = cell_mask & ~nuc_mask & ~adaptive_ring

        membrane_dab_list.append(float(dab[adaptive_ring].mean()))
        nucleus_dab_list.append(float(dab[nuc_mask].mean()))
        if cyto_mask.sum() > 3:
            cytoplasm_dab_list.append(float(dab[cyto_mask].mean()))

    return {
        "membrane_dab": membrane_dab_list,
        "cytoplasm_dab": cytoplasm_dab_list,
        "nucleus_dab": nucleus_dab_list,
        "peak_distances": peak_distances,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. CLINICAL H-SCORE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


def classify_dab_intensity(dab_value: float, thresholds: tuple[float, float, float]) -> int:
    """Classify DAB optical density into 0/1+/2+/3+."""
    t1, t2, t3 = thresholds
    if dab_value < t1:
        return 0
    elif dab_value < t2:
        return 1
    elif dab_value < t3:
        return 2
    else:
        return 3


def compute_hscore(classifications: list[int]) -> float:
    """Compute H-score: H = 1×%1+ + 2×%2+ + 3×%3+. Range 0–300."""
    if not classifications:
        return 0.0
    n = len(classifications)
    c = np.array(classifications)
    pct_1 = (c == 1).sum() / n * 100
    pct_2 = (c == 2).sum() / n * 100
    pct_3 = (c == 3).sum() / n * 100
    return float(1 * pct_1 + 2 * pct_2 + 3 * pct_3)


def compute_cldn18_positivity(classifications: list[int]) -> float:
    """CLDN18.2 positivity: % cells with 2+ or 3+ staining."""
    if not classifications:
        return 0.0
    c = np.array(classifications)
    return float(((c >= 2).sum() / len(c)) * 100)


def measure_hscore_both_methods(
    dab: np.ndarray,
    pred_nuc: np.ndarray,
    pred_cell: np.ndarray,
    thresholds: tuple[float, float, float],
    expansion_px: int = 10,
    min_cell_area: int = 20,
) -> dict:
    """Compute H-score using both model boundaries and nucleus expansion.

    Returns per-cell classifications and tile-level H-scores for both methods.
    """
    model_classes = []
    expansion_classes = []
    model_membrane_dabs = []
    expansion_membrane_dabs = []

    # Create expansion masks from predicted nuclei
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * expansion_px + 1, 2 * expansion_px + 1)
    )

    for cell_id in np.unique(pred_cell):
        if cell_id == 0:
            continue

        cell_mask = pred_cell == cell_id
        nuc_mask = pred_nuc == cell_id

        if cell_mask.sum() < min_cell_area or not nuc_mask.any():
            continue

        # --- Method A: Model boundaries with 10px membrane ring ---
        cell_u8 = cell_mask.astype(np.uint8)
        eroded = cv2.erode(cell_u8, np.ones((5, 5), np.uint8), iterations=2)
        model_ring = cell_mask & ~eroded.astype(bool)

        if model_ring.sum() < 3:
            continue

        model_membrane_dab = float(dab[model_ring].mean())
        model_membrane_dabs.append(model_membrane_dab)
        model_classes.append(classify_dab_intensity(model_membrane_dab, thresholds))

        # --- Method B: 5μm nucleus expansion ring ---
        nuc_u8 = nuc_mask.astype(np.uint8)
        expanded = cv2.dilate(nuc_u8, kernel, iterations=1).astype(bool)
        # Ring = expanded minus nucleus
        expansion_ring = expanded & ~nuc_mask

        if expansion_ring.sum() < 3:
            expansion_membrane_dabs.append(model_membrane_dab)
            expansion_classes.append(classify_dab_intensity(model_membrane_dab, thresholds))
            continue

        exp_membrane_dab = float(dab[expansion_ring].mean())
        expansion_membrane_dabs.append(exp_membrane_dab)
        expansion_classes.append(classify_dab_intensity(exp_membrane_dab, thresholds))

    return {
        "model": {
            "classes": model_classes,
            "hscore": compute_hscore(model_classes),
            "cldn18_positivity": compute_cldn18_positivity(model_classes),
            "membrane_dabs": model_membrane_dabs,
        },
        "expansion": {
            "classes": expansion_classes,
            "hscore": compute_hscore(expansion_classes),
            "cldn18_positivity": compute_cldn18_positivity(expansion_classes),
            "membrane_dabs": expansion_membrane_dabs,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. FIXED RING MEASUREMENT (baseline comparison)
# ═══════════════════════════════════════════════════════════════════════════


def measure_fixed_ring_per_cell(
    dab: np.ndarray,
    cell_labels: np.ndarray,
    nuc_labels: np.ndarray,
    min_cell_area: int = 20,
) -> dict:
    """Standard fixed 10px ring measurement for baseline comparison."""
    membrane_dab_list = []
    cytoplasm_dab_list = []
    nucleus_dab_list = []

    for cell_id in np.unique(cell_labels):
        if cell_id == 0:
            continue

        cell_mask = cell_labels == cell_id
        nuc_mask = nuc_labels == cell_id

        if cell_mask.sum() < min_cell_area or not nuc_mask.any():
            continue

        cell_u8 = cell_mask.astype(np.uint8)
        eroded = cv2.erode(cell_u8, np.ones((5, 5), np.uint8), iterations=2)
        membrane_ring = cell_mask & ~eroded.astype(bool)
        cyto_mask = cell_mask & ~nuc_mask & ~membrane_ring

        if membrane_ring.sum() < 5:
            continue

        membrane_dab_list.append(float(dab[membrane_ring].mean()))
        nucleus_dab_list.append(float(dab[nuc_mask].mean()))
        if cyto_mask.sum() > 3:
            cytoplasm_dab_list.append(float(dab[cyto_mask].mean()))

    return {
        "membrane_dab": membrane_dab_list,
        "cytoplasm_dab": cytoplasm_dab_list,
        "nucleus_dab": nucleus_dab_list,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING + INFERENCE
# ═══════════════════════════════════════════════════════════════════════════


def load_model_and_postprocess(cfg: dict, device: str = "cuda:0"):
    """Load trained InstanSeg model + pixel classifier + loss module."""
    from instanseg.utils.model_loader import build_model_from_dict
    from instanseg.utils.loss.instanseg_loss import InstanSeg as InstanSegLoss, ProbabilityNet
    from instanseg.utils.augmentations import Augmentations

    tcfg = cfg["training"]
    ckpt = torch.load(
        "/home/fernandosoto/Documents/models/brightfield_cells_nuclei/model_weights.pth",
        weights_only=False, map_location=device,
    )
    model = build_model_from_dict({
        "model_str": "InstanSeg_UNet", "layers": tuple(tcfg["layers"]),
        "dim_in": 3, "n_sigma": tcfg["n_sigma"], "dim_coords": tcfg["dim_coords"],
        "dim_seeds": tcfg["dim_seeds"], "norm": tcfg["norm"],
        "cells_and_nuclei": True, "multihead": False, "dropprob": 0.0,
    })
    loadable = {k: v for k, v in ckpt["model_state_dict"].items() if k in model.state_dict()}
    model.load_state_dict(loadable, strict=False)
    model.eval().to(device)

    pc = ProbabilityNet(embedding_dim=tcfg["dim_coords"] + tcfg["n_sigma"] - 2 + 2, width=tcfg["mlp_width"])
    pc_sd = {k.replace("pixel_classifier.", ""): v
             for k, v in ckpt["model_state_dict"].items() if "pixel_classifier" in k}
    if pc_sd:
        pc.load_state_dict(pc_sd, strict=False)
    pc.eval().to(device)

    loss_fn = InstanSegLoss(n_sigma=tcfg["n_sigma"], dim_coords=tcfg["dim_coords"],
                            dim_seeds=tcfg["dim_seeds"], cells_and_nuclei=True, window_size=32)
    loss_fn.pixel_classifier = pc
    loss_fn.eval().to(device)

    Aug = Augmentations()
    return model, loss_fn, Aug


def predict_tile(model, loss_fn, Aug, img: np.ndarray, device: str = "cuda:0"):
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
# CALIBRATE H-SCORE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════


def calibrate_thresholds(all_membrane_dabs: list[float]) -> tuple[float, float, float]:
    """Calibrate DAB intensity thresholds from data distribution.

    Uses Otsu-style multi-threshold or fixed percentile-based thresholds.
    Falls back to standard IHC thresholds if data is insufficient.
    """
    if len(all_membrane_dabs) < 50:
        # Standard IHC thresholds for DAB optical density
        return (0.10, 0.20, 0.35)

    arr = np.array(all_membrane_dabs)

    # Use quartile-based thresholds: ~25% in each class
    # This is common in IHC scoring when no pathologist calibration available
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])

    # Sanity: ensure thresholds are reasonable (DAB OD typically 0-0.5)
    t1 = max(0.05, min(q25, 0.15))   # Neg/1+ boundary
    t2 = max(0.10, min(q50, 0.25))   # 1+/2+ boundary
    t3 = max(0.20, min(q75, 0.40))   # 2+/3+ boundary

    return (t1, t2, t3)


# ═══════════════════════════════════════════════════════════════════════════
# CONCORDANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════


def compute_concordance_stats(
    model_hscores: list[float],
    expansion_hscores: list[float],
) -> dict:
    """Compute concordance metrics between model and expansion H-scores."""
    if len(model_hscores) < 5:
        return {"error": "insufficient data"}

    m = np.array(model_hscores)
    e = np.array(expansion_hscores)

    # Paired statistics
    diff = m - e
    stats = {
        "n_tiles": len(m),
        "model_hscore_mean": float(m.mean()),
        "model_hscore_std": float(m.std()),
        "expansion_hscore_mean": float(e.mean()),
        "expansion_hscore_std": float(e.std()),
        "mean_difference": float(diff.mean()),
        "std_difference": float(diff.std()),
        "model_higher_count": int((diff > 0).sum()),
        "expansion_higher_count": int((diff < 0).sum()),
        "equal_count": int((diff == 0).sum()),
    }

    # Pearson correlation
    if m.std() > 0 and e.std() > 0:
        stats["pearson_r"] = float(np.corrcoef(m, e)[0, 1])

    # Intraclass correlation (ICC(3,1) — two-way mixed, consistency)
    n = len(m)
    if n > 2:
        grand_mean = (m.mean() + e.mean()) / 2
        ssb = n * ((m.mean() - grand_mean) ** 2 + (e.mean() - grand_mean) ** 2)
        ssw = ((m - m.mean()) ** 2).sum() + ((e - e.mean()) ** 2).sum()
        sse = ((m - e) ** 2).sum() / 2
        msr = ssb / 1 if ssb > 0 else 0
        mse = sse / (n - 1) if n > 1 else 0
        msw = ssw / n if n > 0 else 0

        # ICC(3,1) = (MSR - MSE) / (MSR + MSE)
        if (msr + mse) > 0:
            stats["icc_31"] = float((msr - mse) / (msr + mse))

    # Paired t-test (is model different from expansion?)
    if n > 2 and diff.std() > 0:
        t_stat = diff.mean() / (diff.std() / np.sqrt(n))
        # Two-sided p-value approximation (normal for large n)
        from scipy.stats import t as t_dist
        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1))
        stats["paired_t_stat"] = float(t_stat)
        stats["paired_p_value"] = float(p_val)

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Comprehensive membrane validation")
    parser.add_argument("--device", default="cuda:0", help="GPU device (use CUDA_VISIBLE_DEVICES to pick GPU)")
    parser.add_argument("--n-tiles", type=int, default=200, help="Number of test tiles")
    parser.add_argument("--config", default=None, help="Config YAML path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = args.device
    n_tiles = args.n_tiles
    t0 = time.time()

    logger.info("=" * 72)
    logger.info("COMPREHENSIVE MEMBRANE VALIDATION")
    logger.info("  Task 1: Aperio bar-filter membrane detection")
    logger.info("  Task 2: Adaptive per-cell ring width")
    logger.info("  Task 3: Clinical H-score concordance")
    logger.info("  Device: %s  |  Tiles: %d", device, n_tiles)
    logger.info("=" * 72)

    # --- Setup ---
    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    eval_dir = Path(cfg["paths"]["evaluation_dir"])
    sc = cfg["stain_deconvolution"]
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))

    # Load model
    logger.info("Loading InstanSeg model on %s...", device)
    model, loss_fn, Aug = load_model_and_postprocess(cfg, device)
    logger.info("Model loaded (%.1fs)", time.time() - t0)

    # Build bar-filter kernels and move to GPU
    bar_kernels = build_bar_kernels(n_orientations=8, ksize=25, sigma_long=5.0, sigma_short=1.0)
    bar_kernels = bar_kernels.to(device)
    logger.info("Bar-filter kernels built: %s", bar_kernels.shape)

    # Load test data
    dataset_path = data_dir / "segmentation_dataset_base.pth"
    if not dataset_path.exists():
        dataset_path = data_dir / "segmentation_dataset.pth"
    dataset = torch.load(dataset_path, weights_only=False)
    test_data = dataset["Test"][:n_tiles]
    logger.info("Test tiles: %d", len(test_data))

    # --- Accumulators ---
    # Task 1: Bar-filter
    all_bar = {"membrane_dab": [], "cytoplasm_dab": [], "nucleus_dab": [], "completeness": []}
    # Task 2: Adaptive ring
    all_adaptive = {"membrane_dab": [], "cytoplasm_dab": [], "nucleus_dab": [], "peak_distances": []}
    # Task 3: H-score
    all_model_hscores = []
    all_expansion_hscores = []
    all_model_positivity = []
    all_expansion_positivity = []
    # Baseline: fixed ring
    all_fixed = {"membrane_dab": [], "cytoplasm_dab": [], "nucleus_dab": []}
    # For threshold calibration
    calibration_dabs = []

    tiles_processed = 0
    total_cells = 0
    skipped = 0

    # --- First pass: collect membrane DABs for threshold calibration ---
    logger.info("Calibration pass: collecting membrane DAB distribution...")
    cal_count = min(50, len(test_data))
    for item in tqdm(test_data[:cal_count], desc="Calibrating"):
        img = tifffile.imread(data_dir / item["image"])
        try:
            pred_nuc, pred_cell = predict_tile(model, loss_fn, Aug, img, device)
        except Exception:
            continue
        if pred_cell.max() == 0:
            continue

        dab = deconv.extract_dab(img)
        for cell_id in np.unique(pred_cell):
            if cell_id == 0:
                continue
            cell_mask = pred_cell == cell_id
            nuc_mask = pred_nuc == cell_id
            if cell_mask.sum() < 20 or not nuc_mask.any():
                continue
            cell_u8 = cell_mask.astype(np.uint8)
            eroded = cv2.erode(cell_u8, np.ones((5, 5), np.uint8), iterations=2)
            ring = cell_mask & ~eroded.astype(bool)
            if ring.sum() > 3:
                calibration_dabs.append(float(dab[ring].mean()))

    thresholds = calibrate_thresholds(calibration_dabs)
    logger.info("H-score thresholds calibrated: Neg<%.3f, 1+<%.3f, 2+<%.3f, 3+>=%.3f",
                thresholds[0], thresholds[1], thresholds[2], thresholds[2])

    # --- Main pass: all 3 tasks ---
    logger.info("Main pass: running all 3 measurement approaches...")
    for item in tqdm(test_data, desc="Validating"):
        img = tifffile.imread(data_dir / item["image"])

        try:
            pred_nuc, pred_cell = predict_tile(model, loss_fn, Aug, img, device)
        except Exception:
            skipped += 1
            continue

        if pred_nuc.max() == 0 or pred_cell.max() == 0:
            skipped += 1
            continue

        dab = deconv.extract_dab(img)

        # --- Task 1: Bar-filter ---
        dab_tensor = torch.from_numpy(dab).float().unsqueeze(0).unsqueeze(0).to(device)
        bar_resp = apply_bar_filters_gpu(dab_tensor, bar_kernels)
        bar_resp_np = bar_resp[0].cpu().numpy()

        bar_results = measure_bar_filter_per_cell(dab, bar_resp_np, pred_cell, pred_nuc)
        for k in all_bar:
            all_bar[k].extend(bar_results[k])

        # --- Task 2: Adaptive ring ---
        adaptive_results = measure_adaptive_ring_per_cell(dab, pred_cell, pred_nuc, n_rays=16)
        for k in all_adaptive:
            all_adaptive[k].extend(adaptive_results[k])

        # --- Task 3: H-score (per tile) ---
        hscore_results = measure_hscore_both_methods(
            dab, pred_nuc, pred_cell, thresholds, expansion_px=10,
        )
        if hscore_results["model"]["classes"]:
            all_model_hscores.append(hscore_results["model"]["hscore"])
            all_expansion_hscores.append(hscore_results["expansion"]["hscore"])
            all_model_positivity.append(hscore_results["model"]["cldn18_positivity"])
            all_expansion_positivity.append(hscore_results["expansion"]["cldn18_positivity"])

        # --- Baseline: fixed ring ---
        fixed_results = measure_fixed_ring_per_cell(dab, pred_cell, pred_nuc)
        for k in all_fixed:
            all_fixed[k].extend(fixed_results[k])

        tiles_processed += 1
        total_cells += pred_cell.max()

    elapsed = time.time() - t0
    logger.info("Processed %d tiles (%d skipped), %d total cells in %.1fs",
                tiles_processed, skipped, total_cells, elapsed)

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════

    def _summarize(name, data):
        m = float(np.mean(data["membrane_dab"])) if data["membrane_dab"] else 0
        c = float(np.mean(data["cytoplasm_dab"])) if data["cytoplasm_dab"] else 0
        n = float(np.mean(data["nucleus_dab"])) if data["nucleus_dab"] else 0
        gap = m - c
        passed = m > c
        return {
            "method": name,
            "n_cells": len(data["membrane_dab"]),
            "membrane_dab_mean": round(m, 4),
            "cytoplasm_dab_mean": round(c, 4),
            "nucleus_dab_mean": round(n, 4),
            "gap": round(gap, 4),
            "test2_pass": passed,
        }

    results_fixed = _summarize("fixed_10px_ring", all_fixed)
    results_bar = _summarize("aperio_bar_filter", all_bar)
    if all_bar["completeness"]:
        results_bar["mean_completeness"] = round(float(np.mean(all_bar["completeness"])), 4)
    results_adaptive = _summarize("adaptive_ring_16ray", all_adaptive)
    if all_adaptive["peak_distances"]:
        results_adaptive["median_peak_distance_px"] = round(float(np.median(all_adaptive["peak_distances"])), 1)

    # H-score concordance
    concordance = compute_concordance_stats(all_model_hscores, all_expansion_hscores)
    concordance["thresholds_used"] = list(thresholds)
    concordance["model_mean_positivity"] = round(float(np.mean(all_model_positivity)), 2) if all_model_positivity else 0
    concordance["expansion_mean_positivity"] = round(float(np.mean(all_expansion_positivity)), 2) if all_expansion_positivity else 0

    # ═══════════════════════════════════════════════════════════════════
    # PRINT RESULTS
    # ═══════════════════════════════════════════════════════════════════

    logger.info("\n" + "=" * 72)
    logger.info("COMPREHENSIVE MEMBRANE VALIDATION — RESULTS")
    logger.info("=" * 72)

    logger.info("\n--- COMPARTMENT INTENSITY (Test 2) ---")
    for r in [results_fixed, results_bar, results_adaptive]:
        status = "PASS" if r["test2_pass"] else "FAIL"
        logger.info("  %-25s  Membrane=%.4f  Cyto=%.4f  Gap=%+.4f  [%s]  (%d cells)",
                     r["method"], r["membrane_dab_mean"], r["cytoplasm_dab_mean"],
                     r["gap"], status, r["n_cells"])

    if "mean_completeness" in results_bar:
        logger.info("  Bar-filter: mean membrane completeness = %.1f%%",
                     results_bar["mean_completeness"] * 100)

    if "median_peak_distance_px" in results_adaptive:
        logger.info("  Adaptive: median peak distance = %.1f px from centroid",
                     results_adaptive["median_peak_distance_px"])

    logger.info("\n--- CLINICAL H-SCORE (Task 3) ---")
    logger.info("  Thresholds: Neg<%.3f, 1+<%.3f, 2+<%.3f, 3+>=%.3f",
                thresholds[0], thresholds[1], thresholds[2], thresholds[2])
    logger.info("  Model boundaries:    H-score=%.1f +/- %.1f   CLDN18.2+=%.1f%%",
                concordance.get("model_hscore_mean", 0),
                concordance.get("model_hscore_std", 0),
                concordance.get("model_mean_positivity", 0))
    logger.info("  Nucleus expansion:   H-score=%.1f +/- %.1f   CLDN18.2+=%.1f%%",
                concordance.get("expansion_hscore_mean", 0),
                concordance.get("expansion_hscore_std", 0),
                concordance.get("expansion_mean_positivity", 0))
    logger.info("  Mean difference:     %+.1f (model - expansion)",
                concordance.get("mean_difference", 0))
    if "pearson_r" in concordance:
        logger.info("  Pearson correlation:  r=%.3f", concordance["pearson_r"])
    if "icc_31" in concordance:
        logger.info("  ICC(3,1):            %.3f", concordance["icc_31"])
    if "paired_p_value" in concordance:
        logger.info("  Paired t-test:       t=%.2f, p=%.4f",
                     concordance["paired_t_stat"], concordance["paired_p_value"])
    logger.info("  Model higher H-score: %d/%d tiles (%.1f%%)",
                concordance.get("model_higher_count", 0),
                concordance.get("n_tiles", 0),
                100 * concordance.get("model_higher_count", 0) / max(concordance.get("n_tiles", 1), 1))

    logger.info("\n" + "=" * 72)
    logger.info("Wall time: %.1fs (%.2fs/tile)", elapsed, elapsed / max(tiles_processed, 1))
    logger.info("=" * 72)

    # ═══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "n_tiles_processed": tiles_processed,
        "n_tiles_skipped": skipped,
        "total_cells": int(total_cells),
        "wall_time_s": round(elapsed, 1),
        "task1_bar_filter": results_bar,
        "task2_adaptive_ring": results_adaptive,
        "task3_hscore_concordance": concordance,
        "baseline_fixed_ring": results_fixed,
    }

    out_dir = eval_dir / "comprehensive_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
