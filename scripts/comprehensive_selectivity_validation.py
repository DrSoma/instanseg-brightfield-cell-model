#!/usr/bin/env python3
"""Comprehensive selectivity validation for brightfield_cells_nuclei InstanSeg model.

Runs 8 independent tests to characterize whether the model is DAB-selective
(detects only CLDN18.2-expressing cells) or a general cell detector:

  Test 1: Visual overlays on purest blue tiles vs high-DAB tiles
  Test 2: DAB measurement at detected boundaries on negative tissue
  Test 3: H&E slide test (cross-stain generalisation check)
  Test 4: Cross-reference with baseline nuclei model
  Test 5: Virchow2 epithelial classification
  Test 6: Fluorescence model comparison
  Test 7: Training mask overlap (GT signal check)
  Test 8: Multi-panel heatmaps (raw model outputs, bar-filter, boundaries)

Usage:
    python scripts/comprehensive_selectivity_validation.py --all
    python scripts/comprehensive_selectivity_validation.py --test 1,2,4
    python scripts/comprehensive_selectivity_validation.py --test 3 --he-tiles 30
"""

from __future__ import annotations

# ---- Environment (must precede any torch / Qt import) ----
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tifffile
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from tqdm import tqdm

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path("/home/fernandosoto/Documents/instanseg-brightfield-cell-model")
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "segmentation_dataset.pth"

VENV_PKG = Path(
    "/home/fernandosoto/claudin18_venv/lib/python3.12/site-packages"
)
OUR_MODEL_PATH = (
    VENV_PKG / "instanseg/bioimageio_models/brightfield_cells_nuclei/instanseg.pt"
)
BASELINE_MODEL_PATH = (
    VENV_PKG / "instanseg/bioimageio_models/brightfield_nuclei/0.1.1/instanseg.pt"
)
FLUORO_MODEL_PATH = (
    VENV_PKG
    / "instanseg/bioimageio_models/fluorescence_nuclei_and_cells/0.1.1/instanseg.pt"
)

HE_SLIDE_PATH = Path(
    "/home/fernandosoto/Downloads/LungMUHC00183(A2-1)HandE.ndpi"
)

OUT_DIR = PROJECT_ROOT / "evaluation" / "selectivity_validation"
OVERLAY_DIR = OUT_DIR / "overlays"
HEATMAP_DIR = OUT_DIR / "heatmaps"
RESULTS_JSON = OUT_DIR / "results.json"

# ============================================================================
# Fixed tile lists (purest blue = mean DAB <= 0.0001)
# ============================================================================
BLUE_TILES = [
    "tiles/CLDN0216/95744_19712.tiff",
    "tiles/CLDN0216/88000_15488.tiff",
    "tiles/CLDN0235/92224_23232.tiff",
    "tiles/CLDN0235/85888_25344.tiff",
    "tiles/CLDN0235/92928_25344.tiff",
    "tiles/CLDN0271/26752_76032.tiff",
    "tiles/CLDN0271/28864_72512.tiff",
    "tiles/CLDN0216/100672_16192.tiff",
    "tiles/CLDN0216/91520_16896.tiff",
    "tiles/CLDN0216/110528_9856.tiff",
]

# ============================================================================
# Stain deconvolution constants (calibrated for CLDN18.2 IHC)
# ============================================================================
_H_VEC = np.array([0.786, 0.593, 0.174], dtype=np.float64)
_DAB_VEC = np.array([0.215, 0.422, 0.881], dtype=np.float64)
_RES_VEC = np.array([0.547, -0.799, 0.249], dtype=np.float64)

_H_VEC /= np.linalg.norm(_H_VEC)
_DAB_VEC /= np.linalg.norm(_DAB_VEC)
_RES_VEC /= np.linalg.norm(_RES_VEC)

_STAIN_MATRIX = np.stack([_H_VEC, _DAB_VEC, _RES_VEC], axis=0)
_DECONV_MATRIX = np.linalg.inv(_STAIN_MATRIX)


# ============================================================================
# Utility helpers
# ============================================================================

def extract_dab(img_uint8: np.ndarray) -> np.ndarray:
    """Return DAB concentration map (H, W) from an RGB uint8 image."""
    img = np.clip(img_uint8.astype(np.float64) / 255.0, 1e-6, 1.0)
    od = -np.log10(img)
    stain_conc = od.reshape(-1, 3) @ _DECONV_MATRIX.T
    dab = np.clip(stain_conc[:, 1].reshape(img_uint8.shape[:2]), 0, None)
    return dab


def tissue_mask_otsu(img_uint8: np.ndarray) -> np.ndarray:
    """Binary mask (True = tissue) via grayscale Otsu thresholding."""
    gray = np.mean(img_uint8.astype(np.float64), axis=2)
    hist, bin_edges = np.histogram(gray, bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = hist.sum()
    sum_total = (hist * bin_centers).sum()
    w0, sum0, max_var, threshold = 0.0, 0.0, 0.0, 128.0
    for i in range(256):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * bin_centers[i]
        mean0 = sum0 / w0
        mean1 = (sum_total - sum0) / w1
        bvar = w0 * w1 * (mean0 - mean1) ** 2
        if bvar > max_var:
            max_var = bvar
            threshold = bin_centers[i]
    return gray < threshold


def img_to_tensor(img_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (H, W, 3) uint8 RGB to (1, 3, H, W) float32 tensor on device."""
    img_f = img_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)


def count_labels(label_map: torch.Tensor) -> int:
    """Count unique non-zero labels in a tensor."""
    u = torch.unique(label_map)
    return int((u > 0).sum().item())


def labels_to_np(label_tensor: torch.Tensor) -> np.ndarray:
    """Convert a label tensor (possibly batched) to a 2-D int32 numpy array."""
    t = label_tensor
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3:
        t = t[0]
    return t.cpu().numpy().astype(np.int32)


def run_our_model(
    model: torch.jit.ScriptModule,
    img_t: torch.Tensor,
    target_cn: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run our cells+nuclei model.  Returns (nuc_labels, cell_labels) as int32."""
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        out = model(img_t, target_segmentation=target_cn)
    # out shape: (1, 2, H, W) — channel 0 = nuclei, channel 1 = cells
    out = out[0]  # (2, H, W)
    nuc = out[0].cpu().numpy().astype(np.int32)
    cell = out[1].cpu().numpy().astype(np.int32)
    return nuc, cell


def run_baseline_model(
    model: torch.jit.ScriptModule,
    img_t: torch.Tensor,
    target_n: torch.Tensor,
) -> np.ndarray:
    """Run the baseline nuclei-only model.  Returns nuclei labels as int32."""
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        out = model(img_t, target_segmentation=target_n)
    return labels_to_np(out)


def run_fluoro_model(
    model: torch.jit.ScriptModule,
    img_t: torch.Tensor,
    target_cn: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the fluorescence cells+nuclei model.  Returns (nuc, cell) int32."""
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        out = model(img_t, target_segmentation=target_cn)
    out = out[0]  # (2, H, W)
    nuc = out[0].cpu().numpy().astype(np.int32)
    cell = out[1].cpu().numpy().astype(np.int32)
    return nuc, cell


def draw_contours(
    canvas: np.ndarray,
    labels: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw instance contours from a label map onto canvas (in-place, RGB)."""
    for lid in np.unique(labels):
        if lid == 0:
            continue
        mask = (labels == lid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, thickness)


def centroids_from_labels(labels: np.ndarray) -> np.ndarray:
    """Return (N, 2) array of [y, x] centroids for each non-zero label."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return np.empty((0, 2), dtype=np.float64)
    centroids = ndi.center_of_mass(labels > 0, labels, ids)
    return np.array(centroids, dtype=np.float64)


# ---- Bar-filter kernels (for Test 8) ----

def build_bar_kernels(
    n_orientations: int = 8,
    ksize: int = 25,
    sigma_long: float = 5.0,
    sigma_short: float = 1.0,
) -> torch.Tensor:
    """Oriented Gaussian bar-filter kernels for membrane detection.

    Returns (n_orientations, 1, ksize, ksize) float32 tensor.
    """
    half = ksize // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
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
    arr = np.stack(kernels)[:, np.newaxis, :, :]
    return torch.from_numpy(arr).float()


def apply_bar_filters(dab_2d: np.ndarray, device: torch.device) -> np.ndarray:
    """Apply bar-filter bank to a DAB map.  Returns max response (H, W)."""
    kernels = build_bar_kernels().to(device)
    dab_t = (
        torch.from_numpy(dab_2d.astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    pad = kernels.shape[-1] // 2
    responses = F.conv2d(dab_t, kernels, padding=pad)
    max_resp, _ = responses.max(dim=1)
    return max_resp[0].cpu().numpy()


# ============================================================================
# Per-tile loader helpers
# ============================================================================

def load_tile(rel_path: str) -> np.ndarray:
    """Load a tile image from DATA_DIR / rel_path.  Returns (H,W,3) uint8 RGB."""
    full = DATA_DIR / rel_path
    if full.suffix == ".tiff":
        return tifffile.imread(str(full))
    return cv2.cvtColor(cv2.imread(str(full)), cv2.COLOR_BGR2RGB)


def find_high_dab_tiles(n: int = 10) -> list[str]:
    """Find n tiles with highest mean DAB from the test set.

    Scans the full test set and returns the top n by mean DAB intensity.
    """
    dataset = torch.load(str(DATASET_PATH), map_location="cpu", weights_only=False)
    test_items = dataset["Test"]

    scored: list[tuple[float, str]] = []
    for item in tqdm(test_items, desc="Scanning DAB levels", leave=False):
        img = load_tile(item["image"])
        dab = extract_dab(img)
        tissue = tissue_mask_otsu(img)
        if tissue.sum() < 100:
            continue
        mean_dab = float(np.mean(dab[tissue]))
        scored.append((mean_dab, item["image"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:n]]


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models(
    device: torch.device,
    need_our: bool = True,
    need_baseline: bool = True,
    need_fluoro: bool = False,
) -> dict[str, torch.jit.ScriptModule | None]:
    """Load requested TorchScript models.  Returns dict keyed by name."""
    models: dict[str, Any] = {
        "our": None,
        "baseline": None,
        "fluoro": None,
    }
    if need_our:
        logger.info("Loading brightfield_cells_nuclei model ...")
        models["our"] = torch.jit.load(str(OUR_MODEL_PATH), map_location=device)
        models["our"].eval()
    if need_baseline:
        logger.info("Loading brightfield_nuclei (baseline) model ...")
        models["baseline"] = torch.jit.load(
            str(BASELINE_MODEL_PATH), map_location=device
        )
        models["baseline"].eval()
    if need_fluoro:
        logger.info("Loading fluorescence_nuclei_and_cells model ...")
        models["fluoro"] = torch.jit.load(
            str(FLUORO_MODEL_PATH), map_location=device
        )
        models["fluoro"].eval()
    return models


# ============================================================================
# TEST 1: Visual Overlays on Blue Tiles
# ============================================================================

def test1_visual_overlays(
    device: torch.device,
    high_dab_tiles: list[str],
) -> dict:
    """Generate overlay images comparing our model vs baseline on blue/DAB tiles.

    Green contours = our model (cell boundaries)
    Red contours = baseline nuclei model
    """
    logger.info("=" * 70)
    logger.info("TEST 1: Visual Overlays on Blue vs DAB-Positive Tiles")
    logger.info("=" * 70)

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    models = load_models(device, need_our=True, need_baseline=True)
    target_cn = torch.tensor([1, 1], dtype=torch.long, device=device)
    target_n = torch.tensor([1], dtype=torch.long, device=device)

    all_tiles = [(p, "blue") for p in BLUE_TILES] + [
        (p, "dab") for p in high_dab_tiles
    ]

    results: list[dict] = []

    for rel_path, category in tqdm(all_tiles, desc="Test 1: Overlays"):
        img = load_tile(rel_path)
        img_t = img_to_tensor(img, device)

        # Our model
        nuc_ours, cell_ours = run_our_model(models["our"], img_t, target_cn)
        our_cell_count = int((np.unique(cell_ours) > 0).sum())
        our_nuc_count = int((np.unique(nuc_ours) > 0).sum())

        # Baseline
        nuc_base = run_baseline_model(models["baseline"], img_t, target_n)
        base_count = int((np.unique(nuc_base) > 0).sum())

        # Build overlay
        overlay = img.copy()
        draw_contours(overlay, cell_ours, color=(0, 255, 0), thickness=2)
        draw_contours(overlay, nuc_base, color=(255, 0, 0), thickness=1)

        # Save
        tile_name = Path(rel_path).stem
        slide_name = Path(rel_path).parent.name
        fname = f"{category}_{slide_name}_{tile_name}.png"
        cv2.imwrite(
            str(OVERLAY_DIR / fname),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

        results.append(
            {
                "tile": rel_path,
                "category": category,
                "our_cells": our_cell_count,
                "our_nuclei": our_nuc_count,
                "baseline_nuclei": base_count,
            }
        )

        # Free VRAM
        del img_t
        torch.cuda.empty_cache()

    # Summary
    blue_our = [r["our_cells"] for r in results if r["category"] == "blue"]
    blue_base = [r["baseline_nuclei"] for r in results if r["category"] == "blue"]
    dab_our = [r["our_cells"] for r in results if r["category"] == "dab"]
    dab_base = [r["baseline_nuclei"] for r in results if r["category"] == "dab"]

    summary = {
        "blue_tiles": len(blue_our),
        "dab_tiles": len(dab_our),
        "blue_our_mean": float(np.mean(blue_our)) if blue_our else 0.0,
        "blue_baseline_mean": float(np.mean(blue_base)) if blue_base else 0.0,
        "dab_our_mean": float(np.mean(dab_our)) if dab_our else 0.0,
        "dab_baseline_mean": float(np.mean(dab_base)) if dab_base else 0.0,
        "overlay_dir": str(OVERLAY_DIR),
        "per_tile": results,
    }

    logger.info(
        "  Blue tiles: our=%.1f cells, baseline=%.1f nuclei (mean)",
        summary["blue_our_mean"],
        summary["blue_baseline_mean"],
    )
    logger.info(
        "  DAB tiles:  our=%.1f cells, baseline=%.1f nuclei (mean)",
        summary["dab_our_mean"],
        summary["dab_baseline_mean"],
    )

    del models
    torch.cuda.empty_cache()
    return summary


# ============================================================================
# TEST 2: DAB Measurement at Detected Boundaries on Negative Tissue
# ============================================================================

def test2_boundary_dab_measurement(device: torch.device) -> dict:
    """Measure actual DAB OD in a 10px ring around detected cells on negative tissue.

    Uses all 485 test tiles, classifies them by DAB level, and compares
    membrane-ring DAB between negative-tissue detections and positive-tissue
    detections.
    """
    logger.info("=" * 70)
    logger.info("TEST 2: DAB Measurement at Detected Boundaries on Negative Tissue")
    logger.info("=" * 70)

    models = load_models(device, need_our=True, need_baseline=False)
    target_cn = torch.tensor([1, 1], dtype=torch.long, device=device)

    dataset = torch.load(str(DATASET_PATH), map_location="cpu", weights_only=False)
    test_items = dataset["Test"]

    RING_WIDTH = 10  # pixels
    DAB_NEG_THRESH = 0.05  # DAB-positive fraction for "negative" tile

    neg_membrane_dab: list[float] = []
    pos_membrane_dab: list[float] = []
    neg_tile_count = 0
    pos_tile_count = 0
    neg_cell_count = 0
    pos_cell_count = 0

    for item in tqdm(test_items, desc="Test 2: Boundary DAB"):
        img = load_tile(item["image"])
        dab = extract_dab(img)
        tissue = tissue_mask_otsu(img)

        if tissue.sum() < 100:
            continue

        dab_in_tissue = dab[tissue]
        dab_pos_frac = float((dab_in_tissue > 0.10).sum()) / tissue.sum()
        is_negative = dab_pos_frac < DAB_NEG_THRESH

        img_t = img_to_tensor(img, device)
        _, cell_labels = run_our_model(models["our"], img_t, target_cn)
        del img_t
        torch.cuda.empty_cache()

        cell_ids = np.unique(cell_labels)
        cell_ids = cell_ids[cell_ids > 0]
        if len(cell_ids) == 0:
            continue

        # For each cell, measure DAB in a 10px ring (dilated boundary minus interior)
        for cid in cell_ids:
            cell_mask = (cell_labels == cid).astype(np.uint8)
            # Erode to get interior
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(cell_mask, kernel, iterations=1)
            # Dilate to get outer ring
            dilated = cv2.dilate(cell_mask, kernel, iterations=RING_WIDTH // 2)
            # Ring = dilated minus eroded (captures membrane zone)
            ring = ((dilated - eroded) > 0).astype(bool)

            if ring.sum() < 5:
                continue

            ring_dab = float(np.mean(dab[ring]))

            if is_negative:
                neg_membrane_dab.append(ring_dab)
            else:
                pos_membrane_dab.append(ring_dab)

        if is_negative:
            neg_tile_count += 1
            neg_cell_count += len(cell_ids)
        else:
            pos_tile_count += 1
            pos_cell_count += len(cell_ids)

    summary = {
        "ring_width_px": RING_WIDTH,
        "dab_neg_threshold": DAB_NEG_THRESH,
        "negative_tiles": neg_tile_count,
        "positive_tiles": pos_tile_count,
        "negative_cells_measured": len(neg_membrane_dab),
        "positive_cells_measured": len(pos_membrane_dab),
        "negative_cells_total": neg_cell_count,
        "positive_cells_total": pos_cell_count,
        "neg_membrane_dab_mean": float(np.mean(neg_membrane_dab))
        if neg_membrane_dab
        else 0.0,
        "neg_membrane_dab_median": float(np.median(neg_membrane_dab))
        if neg_membrane_dab
        else 0.0,
        "neg_membrane_dab_std": float(np.std(neg_membrane_dab))
        if neg_membrane_dab
        else 0.0,
        "pos_membrane_dab_mean": float(np.mean(pos_membrane_dab))
        if pos_membrane_dab
        else 0.0,
        "pos_membrane_dab_median": float(np.median(pos_membrane_dab))
        if pos_membrane_dab
        else 0.0,
        "pos_membrane_dab_std": float(np.std(pos_membrane_dab))
        if pos_membrane_dab
        else 0.0,
    }

    logger.info(
        "  Negative tissue: %d cells across %d tiles, mean membrane DAB = %.4f",
        len(neg_membrane_dab),
        neg_tile_count,
        summary["neg_membrane_dab_mean"],
    )
    logger.info(
        "  Positive tissue: %d cells across %d tiles, mean membrane DAB = %.4f",
        len(pos_membrane_dab),
        pos_tile_count,
        summary["pos_membrane_dab_mean"],
    )

    if neg_membrane_dab and pos_membrane_dab:
        ratio = summary["neg_membrane_dab_mean"] / (
            summary["pos_membrane_dab_mean"] + 1e-8
        )
        summary["neg_vs_pos_dab_ratio"] = round(ratio, 4)
        logger.info("  Ratio (neg/pos) = %.4f", ratio)
        if ratio > 0.7:
            summary["interpretation"] = (
                "Cells on negative tissue have similar membrane DAB to positive tissue "
                "-- these may be sub-threshold DAB-positive cells."
            )
        else:
            summary["interpretation"] = (
                "Cells on negative tissue have lower membrane DAB -- model detects "
                "real cells but they lack strong DAB staining."
            )

    del models
    torch.cuda.empty_cache()
    return summary


# ============================================================================
# TEST 3: H&E Slide Test
# ============================================================================

def test3_he_slide(
    device: torch.device,
    n_tiles: int = 50,
) -> dict:
    """Extract random tissue tiles from an H&E slide and run both models.

    If our model detects many cells on H&E (no DAB present), it is NOT
    DAB-selective.
    """
    logger.info("=" * 70)
    logger.info("TEST 3: H&E Slide Test (cross-stain generalisation)")
    logger.info("=" * 70)

    import openslide

    if not HE_SLIDE_PATH.exists():
        logger.warning("H&E slide not found at %s -- skipping test 3", HE_SLIDE_PATH)
        return {"error": f"H&E slide not found: {HE_SLIDE_PATH}"}

    slide = openslide.OpenSlide(str(HE_SLIDE_PATH))

    # Determine the best level for ~0.5 um/px
    # NDPI stores MPP in properties
    mpp_x = float(slide.properties.get("openslide.mpp-x", "0"))
    mpp_y = float(slide.properties.get("openslide.mpp-y", "0"))

    if mpp_x <= 0:
        # Fall back: assume level 0 is 0.25 um/px (common for 40x NDPI)
        mpp_x = 0.25
        mpp_y = 0.25

    target_mpp = 0.5
    best_level = 0
    best_mpp = mpp_x

    for lvl in range(slide.level_count):
        downsample = slide.level_downsamples[lvl]
        lvl_mpp = mpp_x * downsample
        if abs(lvl_mpp - target_mpp) < abs(best_mpp - target_mpp):
            best_mpp = lvl_mpp
            best_level = lvl

    logger.info(
        "  Slide: %s  levels=%d  level0 MPP=%.4f  using level %d (MPP=%.4f)",
        HE_SLIDE_PATH.name,
        slide.level_count,
        mpp_x,
        best_level,
        best_mpp,
    )

    level_dims = slide.level_dimensions[best_level]
    downsample = slide.level_downsamples[best_level]
    tile_size = 512

    # Find tissue regions via thumbnail
    thumb_scale = 64
    thumb_size = (
        max(1, level_dims[0] // thumb_scale),
        max(1, level_dims[1] // thumb_scale),
    )
    thumb = slide.get_thumbnail(thumb_size)
    thumb_np = np.array(thumb.convert("RGB"))
    gray = np.mean(thumb_np.astype(np.float64), axis=2)
    tissue_thumb = gray < 220  # crude tissue threshold

    # Sample random tissue locations
    tissue_ys, tissue_xs = np.where(tissue_thumb)
    if len(tissue_ys) == 0:
        slide.close()
        return {"error": "No tissue found in H&E thumbnail"}

    rng = np.random.default_rng(42)
    n_candidates = min(len(tissue_ys), n_tiles * 10)
    indices = rng.choice(len(tissue_ys), size=n_candidates, replace=False)

    models = load_models(device, need_our=True, need_baseline=True)
    target_cn = torch.tensor([1, 1], dtype=torch.long, device=device)
    target_n = torch.tensor([1], dtype=torch.long, device=device)

    results: list[dict] = []
    he_overlay_dir = OVERLAY_DIR / "he"
    he_overlay_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(indices, desc="Test 3: H&E tiles"):
        if len(results) >= n_tiles:
            break

        # Convert thumbnail coords to level-0 coords
        ty, tx = tissue_ys[idx], tissue_xs[idx]
        level0_x = int(tx * thumb_scale * downsample)
        level0_y = int(ty * thumb_scale * downsample)

        # Read tile at best_level
        try:
            region = slide.read_region(
                (level0_x, level0_y), best_level, (tile_size, tile_size)
            )
        except Exception:
            continue

        img = np.array(region.convert("RGB"))

        # Skip mostly-white tiles
        if np.mean(img) > 220:
            continue

        # Check tissue content
        tissue_frac = float((np.mean(img.astype(float), axis=2) < 220).mean())
        if tissue_frac < 0.3:
            continue

        img_t = img_to_tensor(img, device)

        # Our model
        _, cell_ours = run_our_model(models["our"], img_t, target_cn)
        our_count = int((np.unique(cell_ours) > 0).sum())

        # Baseline
        nuc_base = run_baseline_model(models["baseline"], img_t, target_n)
        base_count = int((np.unique(nuc_base) > 0).sum())

        del img_t
        torch.cuda.empty_cache()

        results.append(
            {
                "tile_idx": len(results),
                "coords_level0": [level0_x, level0_y],
                "our_cells": our_count,
                "baseline_nuclei": base_count,
                "tissue_fraction": round(tissue_frac, 3),
            }
        )

        # Save first 10 overlays
        if len(results) <= 10:
            overlay = img.copy()
            draw_contours(overlay, cell_ours, color=(0, 255, 0), thickness=2)
            draw_contours(overlay, nuc_base, color=(255, 0, 0), thickness=1)
            cv2.imwrite(
                str(he_overlay_dir / f"he_tile_{len(results):03d}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

    slide.close()

    our_counts = [r["our_cells"] for r in results]
    base_counts = [r["baseline_nuclei"] for r in results]

    summary = {
        "slide": str(HE_SLIDE_PATH),
        "level_used": best_level,
        "mpp": round(best_mpp, 4),
        "tile_size": tile_size,
        "n_tiles": len(results),
        "our_mean_cells": round(float(np.mean(our_counts)), 1)
        if our_counts
        else 0.0,
        "our_median_cells": float(np.median(our_counts)) if our_counts else 0.0,
        "our_zero_tiles": int(sum(1 for c in our_counts if c == 0)),
        "baseline_mean_nuclei": round(float(np.mean(base_counts)), 1)
        if base_counts
        else 0.0,
        "baseline_median_nuclei": float(np.median(base_counts))
        if base_counts
        else 0.0,
        "per_tile": results,
    }

    if our_counts and base_counts:
        mean_our = np.mean(our_counts)
        mean_base = np.mean(base_counts)
        ratio = mean_our / (mean_base + 1e-8)
        summary["ratio"] = round(float(ratio), 4)

        if ratio < 0.05:
            summary["verdict"] = "STRONGLY DAB-SELECTIVE (near-zero H&E detections)"
        elif ratio < 0.20:
            summary["verdict"] = "MOSTLY DAB-SELECTIVE (few H&E detections)"
        elif ratio < 0.50:
            summary["verdict"] = "PARTIALLY DAB-SELECTIVE (moderate H&E detections)"
        else:
            summary["verdict"] = "NOT DAB-SELECTIVE (detects H&E cells freely)"

    logger.info(
        "  H&E: our=%.1f cells/tile, baseline=%.1f nuclei/tile, ratio=%.4f",
        summary.get("our_mean_cells", 0),
        summary.get("baseline_mean_nuclei", 0),
        summary.get("ratio", 0),
    )
    logger.info("  Verdict: %s", summary.get("verdict", "N/A"))

    del models
    torch.cuda.empty_cache()
    return summary


# ============================================================================
# TEST 4: Cross-Reference with Baseline
# ============================================================================

def test4_cross_reference(device: torch.device) -> dict:
    """For each cell detected by our model on blue tiles, check if baseline
    also detects a nucleus within 10px.

    High match rate = our model detects real cells (same ones baseline finds).
    Low match rate = our model detects different structures (possibly artifacts).
    """
    logger.info("=" * 70)
    logger.info("TEST 4: Cross-Reference with Baseline Nuclei Model")
    logger.info("=" * 70)

    MATCH_DISTANCE = 10.0  # pixels

    models = load_models(device, need_our=True, need_baseline=True)
    target_cn = torch.tensor([1, 1], dtype=torch.long, device=device)
    target_n = torch.tensor([1], dtype=torch.long, device=device)

    total_our = 0
    total_matched = 0
    per_tile: list[dict] = []

    for rel_path in tqdm(BLUE_TILES, desc="Test 4: Cross-reference"):
        img = load_tile(rel_path)
        img_t = img_to_tensor(img, device)

        _, cell_ours = run_our_model(models["our"], img_t, target_cn)
        nuc_base = run_baseline_model(models["baseline"], img_t, target_n)

        del img_t
        torch.cuda.empty_cache()

        our_centroids = centroids_from_labels(cell_ours)
        base_centroids = centroids_from_labels(nuc_base)

        n_our = len(our_centroids)
        matched = 0

        if n_our > 0 and len(base_centroids) > 0:
            tree = cKDTree(base_centroids)
            dists, _ = tree.query(our_centroids, k=1)
            matched = int((dists < MATCH_DISTANCE).sum())

        total_our += n_our
        total_matched += matched

        per_tile.append(
            {
                "tile": rel_path,
                "our_cells": n_our,
                "baseline_nuclei": len(base_centroids),
                "matched": matched,
                "match_pct": round(100.0 * matched / n_our, 1) if n_our > 0 else 0.0,
            }
        )

    match_pct = 100.0 * total_matched / total_our if total_our > 0 else 0.0

    summary = {
        "match_distance_px": MATCH_DISTANCE,
        "total_our_cells": total_our,
        "total_matched": total_matched,
        "overall_match_pct": round(match_pct, 1),
        "per_tile": per_tile,
    }

    if match_pct > 80:
        summary["interpretation"] = (
            "High match rate: our model detects the same real cells that "
            "the baseline nuclei model finds."
        )
    elif match_pct > 50:
        summary["interpretation"] = (
            "Moderate match: majority of our detections correspond to real nuclei, "
            "but some may be extra structures."
        )
    else:
        summary["interpretation"] = (
            "Low match rate: our model detects different structures than "
            "the baseline nuclei model."
        )

    logger.info(
        "  %d / %d our cells matched baseline nuclei (%.1f%%)",
        total_matched,
        total_our,
        match_pct,
    )
    logger.info("  %s", summary["interpretation"])

    del models
    torch.cuda.empty_cache()
    return summary


# ============================================================================
# TEST 5: Virchow2 Epithelial Classification
# ============================================================================

def test5_virchow2_epithelial(
    device: torch.device,
    high_dab_tiles: list[str],
) -> dict:
    """Cluster tiles by Virchow2 embeddings, identify the epithelial cluster
    via DAB intensity, and report what fraction of our detections on blue
    tissue are in the epithelial cluster.
    """
    logger.info("=" * 70)
    logger.info("TEST 5: Virchow2 Epithelial Classification")
    logger.info("=" * 70)

    try:
        import timm
        from PIL import Image
        from sklearn.cluster import KMeans
    except ImportError as exc:
        logger.warning("Missing dependency for test 5: %s", exc)
        return {"error": str(exc)}

    # Load Virchow2
    logger.info("  Loading Virchow2 ...")
    try:
        virchow2 = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        virchow2.eval().to(device)
        transform_cfg = timm.data.resolve_data_config(virchow2.pretrained_cfg)
        transform = timm.data.create_transform(
            **transform_cfg, is_training=False
        )
    except Exception as exc:
        logger.warning("Failed to load Virchow2: %s", exc)
        return {"error": f"Virchow2 load failed: {exc}"}

    # Also load our model for cell counts on blue tiles
    models = load_models(device, need_our=True, need_baseline=False)
    target_cn = torch.tensor([1, 1], dtype=torch.long, device=device)

    # Combine blue + DAB tiles
    all_tiles = [(p, "blue") for p in BLUE_TILES] + [
        (p, "dab") for p in high_dab_tiles
    ]

    embeddings: list[np.ndarray] = []
    categories: list[str] = []
    dab_means: list[float] = []
    cell_counts: list[int] = []

    for rel_path, cat in tqdm(all_tiles, desc="Test 5: Virchow2 embeddings"):
        img = load_tile(rel_path)

        # DAB
        dab = extract_dab(img)
        tissue = tissue_mask_otsu(img)
        dab_mean = float(np.mean(dab[tissue])) if tissue.sum() > 100 else 0.0

        # Cell count from our model
        img_t = img_to_tensor(img, device)
        _, cell_labels = run_our_model(models["our"], img_t, target_cn)
        n_cells = int((np.unique(cell_labels) > 0).sum())
        del img_t
        torch.cuda.empty_cache()

        # Virchow2 embedding
        pil_img = Image.fromarray(img)
        tensor_v = transform(pil_img).unsqueeze(0).to(device)

        with torch.inference_mode(), torch.amp.autocast("cuda"):
            output = virchow2(tensor_v)
            if output.ndim == 3:
                cls_tok = output[:, 0, :]
                patch_mean = output[:, 1:, :].mean(dim=1)
                emb = torch.cat([cls_tok, patch_mean], dim=1)
            else:
                emb = output

        embeddings.append(emb.cpu().numpy().astype(np.float32).flatten())
        categories.append(cat)
        dab_means.append(dab_mean)
        cell_counts.append(n_cells)

    del virchow2
    del models
    torch.cuda.empty_cache()

    # KMeans(K=2)
    emb_array = np.stack(embeddings)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(emb_array)

    # Identify epithelial cluster via DAB intensity
    cluster_dab = {}
    for cl in [0, 1]:
        cl_dab = [dab_means[i] for i in range(len(labels)) if labels[i] == cl]
        cluster_dab[cl] = float(np.mean(cl_dab)) if cl_dab else 0.0

    epi_cluster = 0 if cluster_dab[0] > cluster_dab[1] else 1
    stroma_cluster = 1 - epi_cluster

    # Report blue-tile detections in epithelial cluster
    blue_in_epi = 0
    blue_total_cells = 0
    per_tile_results: list[dict] = []

    for i, (rel_path, cat) in enumerate(all_tiles):
        is_epi = labels[i] == epi_cluster
        rec = {
            "tile": rel_path,
            "category": cat,
            "cluster": int(labels[i]),
            "is_epithelial": is_epi,
            "dab_mean": round(dab_means[i], 4),
            "cell_count": cell_counts[i],
        }
        per_tile_results.append(rec)

        if cat == "blue":
            blue_total_cells += cell_counts[i]
            if is_epi:
                blue_in_epi += cell_counts[i]

    blue_epi_tiles = sum(
        1
        for i, (_, cat) in enumerate(all_tiles)
        if cat == "blue" and labels[i] == epi_cluster
    )
    blue_total_tiles = sum(1 for _, cat in all_tiles if cat == "blue")

    summary = {
        "n_tiles": len(all_tiles),
        "embedding_dim": emb_array.shape[1],
        "epithelial_cluster": epi_cluster,
        "cluster_dab_means": {
            f"cluster_{k}": round(v, 4) for k, v in cluster_dab.items()
        },
        "blue_tiles_in_epithelial_cluster": blue_epi_tiles,
        "blue_tiles_total": blue_total_tiles,
        "blue_epi_tile_pct": round(
            100.0 * blue_epi_tiles / blue_total_tiles, 1
        )
        if blue_total_tiles > 0
        else 0.0,
        "blue_cells_in_epi": blue_in_epi,
        "blue_cells_total": blue_total_cells,
        "blue_cells_epi_pct": round(
            100.0 * blue_in_epi / blue_total_cells, 1
        )
        if blue_total_cells > 0
        else 0.0,
        "per_tile": per_tile_results,
    }

    logger.info(
        "  Epithelial cluster: %d (mean DAB=%.4f), Stromal cluster: %d (mean DAB=%.4f)",
        epi_cluster,
        cluster_dab[epi_cluster],
        stroma_cluster,
        cluster_dab[stroma_cluster],
    )
    logger.info(
        "  Blue tiles in epithelial cluster: %d / %d (%.1f%%)",
        blue_epi_tiles,
        blue_total_tiles,
        summary["blue_epi_tile_pct"],
    )

    return summary


# ============================================================================
# TEST 6: Fluorescence Model Comparison
# ============================================================================

def test6_fluorescence_comparison(
    device: torch.device,
    high_dab_tiles: list[str],
) -> dict:
    """Compare cell counts: our model vs fluorescence model vs baseline.

    The fluorescence model is a general cell detector.  If it also detects
    cells on blue tissue, those are real cells.
    """
    logger.info("=" * 70)
    logger.info("TEST 6: Fluorescence Model Comparison")
    logger.info("=" * 70)

    models = load_models(
        device, need_our=True, need_baseline=True, need_fluoro=True
    )
    target_cn = torch.tensor([1, 1], dtype=torch.long, device=device)
    target_n = torch.tensor([1], dtype=torch.long, device=device)

    all_tiles = [(p, "blue") for p in BLUE_TILES] + [
        (p, "dab") for p in high_dab_tiles
    ]

    per_tile: list[dict] = []

    for rel_path, cat in tqdm(all_tiles, desc="Test 6: Fluoro comparison"):
        img = load_tile(rel_path)
        img_t = img_to_tensor(img, device)

        # Our model
        _, cell_ours = run_our_model(models["our"], img_t, target_cn)
        our_count = int((np.unique(cell_ours) > 0).sum())

        # Baseline nuclei
        nuc_base = run_baseline_model(models["baseline"], img_t, target_n)
        base_count = int((np.unique(nuc_base) > 0).sum())

        # Fluorescence model
        _, cell_fluoro = run_fluoro_model(models["fluoro"], img_t, target_cn)
        fluoro_count = int((np.unique(cell_fluoro) > 0).sum())

        del img_t
        torch.cuda.empty_cache()

        per_tile.append(
            {
                "tile": rel_path,
                "category": cat,
                "our_cells": our_count,
                "baseline_nuclei": base_count,
                "fluoro_cells": fluoro_count,
            }
        )

    # Aggregate
    blue_results = [r for r in per_tile if r["category"] == "blue"]
    dab_results = [r for r in per_tile if r["category"] == "dab"]

    def _stats(records: list[dict], key: str) -> dict:
        vals = [r[key] for r in records]
        if not vals:
            return {"mean": 0.0, "median": 0.0, "total": 0}
        return {
            "mean": round(float(np.mean(vals)), 1),
            "median": float(np.median(vals)),
            "total": int(sum(vals)),
        }

    summary = {
        "blue_tiles": len(blue_results),
        "dab_tiles": len(dab_results),
        "blue": {
            "our": _stats(blue_results, "our_cells"),
            "baseline": _stats(blue_results, "baseline_nuclei"),
            "fluoro": _stats(blue_results, "fluoro_cells"),
        },
        "dab": {
            "our": _stats(dab_results, "our_cells"),
            "baseline": _stats(dab_results, "baseline_nuclei"),
            "fluoro": _stats(dab_results, "fluoro_cells"),
        },
        "per_tile": per_tile,
    }

    # Interpretation
    blue_our_mean = summary["blue"]["our"]["mean"]
    blue_fluoro_mean = summary["blue"]["fluoro"]["mean"]
    blue_base_mean = summary["blue"]["baseline"]["mean"]

    if blue_fluoro_mean > 0:
        our_vs_fluoro = blue_our_mean / (blue_fluoro_mean + 1e-8)
    else:
        our_vs_fluoro = 0.0

    summary["blue_our_vs_fluoro_ratio"] = round(our_vs_fluoro, 4)

    if blue_fluoro_mean > blue_base_mean * 0.5:
        summary["fluoro_validates_real_cells"] = True
        summary["interpretation"] = (
            f"Fluorescence model detects {blue_fluoro_mean:.0f} cells/tile on blue "
            f"tissue (vs baseline {blue_base_mean:.0f} nuclei), confirming real cells "
            f"exist. Our model detects {blue_our_mean:.0f} ({our_vs_fluoro:.1%} of "
            f"fluorescence)."
        )
    else:
        summary["fluoro_validates_real_cells"] = False
        summary["interpretation"] = (
            f"Fluorescence model detects few cells ({blue_fluoro_mean:.0f}/tile) "
            f"on blue tissue."
        )

    logger.info("  Blue tissue:")
    logger.info("    Our model:    %.1f cells/tile", blue_our_mean)
    logger.info("    Fluorescence: %.1f cells/tile", blue_fluoro_mean)
    logger.info("    Baseline:     %.1f nuclei/tile", blue_base_mean)
    logger.info("  %s", summary["interpretation"])

    del models
    torch.cuda.empty_cache()
    return summary


# ============================================================================
# TEST 7: Training Mask Overlap
# ============================================================================

def test7_training_mask_overlap(device: torch.device) -> dict:
    """For the 10 blue tiles, check if the GT masks contain any cell labels.

    If GT has zero cells but our model detects cells, the model generalised
    beyond training data.  If GT also has cells, the tiles had training signal
    (may not be truly "negative").
    """
    logger.info("=" * 70)
    logger.info("TEST 7: Training Mask Overlap (GT signal check)")
    logger.info("=" * 70)

    # Load dataset to find matching GT paths
    dataset = torch.load(str(DATASET_PATH), map_location="cpu", weights_only=False)
    test_items = dataset["Test"]

    # Build lookup: image path -> item
    path_to_item: dict[str, dict] = {}
    for item in test_items:
        path_to_item[item["image"]] = item

    models = load_models(device, need_our=True, need_baseline=False)
    target_cn = torch.tensor([1, 1], dtype=torch.long, device=device)

    per_tile: list[dict] = []

    for rel_path in tqdm(BLUE_TILES, desc="Test 7: GT mask overlap"):
        img = load_tile(rel_path)
        img_t = img_to_tensor(img, device)

        _, cell_ours = run_our_model(models["our"], img_t, target_cn)
        our_count = int((np.unique(cell_ours) > 0).sum())

        del img_t
        torch.cuda.empty_cache()

        # Find GT masks
        item = path_to_item.get(rel_path)

        gt_nuc_count = 0
        gt_cell_count = 0
        gt_nuc_path = None
        gt_cell_path = None

        if item:
            # Nucleus GT
            nuc_mask_path = DATA_DIR / item["nucleus_masks"]
            if nuc_mask_path.exists():
                gt_nuc_path = str(nuc_mask_path)
                nuc_mask = tifffile.imread(str(nuc_mask_path))
                gt_nuc_count = int(len(np.unique(nuc_mask)) - 1)
                gt_nuc_count = max(gt_nuc_count, 0)

            # Cell GT (teacher predictions)
            cell_mask_path = DATA_DIR / item["cell_masks"]
            if cell_mask_path.exists():
                gt_cell_path = str(cell_mask_path)
                cell_mask = tifffile.imread(str(cell_mask_path))
                gt_cell_count = int(len(np.unique(cell_mask)) - 1)
                gt_cell_count = max(gt_cell_count, 0)
        else:
            # Try manual path construction
            parts = Path(rel_path)
            slide = parts.parent.name
            stem = parts.stem

            nuc_path = DATA_DIR / "masks" / slide / f"{stem}_nuclei.tiff"
            cell_path = DATA_DIR / "masks" / slide / f"{stem}_cells.tiff"

            if nuc_path.exists():
                gt_nuc_path = str(nuc_path)
                nuc_mask = tifffile.imread(str(nuc_path))
                gt_nuc_count = max(0, int(len(np.unique(nuc_mask)) - 1))
            if cell_path.exists():
                gt_cell_path = str(cell_path)
                cell_mask = tifffile.imread(str(cell_path))
                gt_cell_count = max(0, int(len(np.unique(cell_mask)) - 1))

        per_tile.append(
            {
                "tile": rel_path,
                "gt_nuclei": gt_nuc_count,
                "gt_cells": gt_cell_count,
                "our_cells": our_count,
                "gt_nuc_path": gt_nuc_path,
                "gt_cell_path": gt_cell_path,
                "in_dataset": item is not None,
            }
        )

    # Aggregate
    tiles_with_gt_cells = sum(1 for r in per_tile if r["gt_cells"] > 0)
    tiles_with_gt_nuclei = sum(1 for r in per_tile if r["gt_nuclei"] > 0)
    tiles_with_our_cells = sum(1 for r in per_tile if r["our_cells"] > 0)
    tiles_gt_zero_our_nonzero = sum(
        1
        for r in per_tile
        if r["gt_cells"] == 0 and r["gt_nuclei"] == 0 and r["our_cells"] > 0
    )

    summary = {
        "n_tiles": len(per_tile),
        "tiles_with_gt_nuclei": tiles_with_gt_nuclei,
        "tiles_with_gt_cells": tiles_with_gt_cells,
        "tiles_with_our_detections": tiles_with_our_cells,
        "tiles_gt_zero_our_nonzero": tiles_gt_zero_our_nonzero,
        "per_tile": per_tile,
    }

    if tiles_gt_zero_our_nonzero > 0:
        summary["interpretation"] = (
            f"{tiles_gt_zero_our_nonzero} / {len(per_tile)} blue tiles have "
            f"ZERO GT labels but our model detects cells -- the model generalised "
            f"beyond its training signal."
        )
    elif tiles_with_gt_cells > 0 or tiles_with_gt_nuclei > 0:
        summary["interpretation"] = (
            f"GT masks have labels on {tiles_with_gt_nuclei} (nuclei) / "
            f"{tiles_with_gt_cells} (cells) of the blue tiles. These tiles "
            f"had training signal and may not be truly negative."
        )
    else:
        summary["interpretation"] = (
            "No GT masks found for blue tiles. Cannot determine if model "
            "generalised or had training signal."
        )

    logger.info("  GT nuclei on blue tiles: %d / %d", tiles_with_gt_nuclei, len(per_tile))
    logger.info("  GT cells on blue tiles:  %d / %d", tiles_with_gt_cells, len(per_tile))
    logger.info("  Our detections:          %d / %d", tiles_with_our_cells, len(per_tile))
    logger.info("  %s", summary["interpretation"])

    del models
    torch.cuda.empty_cache()
    return summary


# ============================================================================
# TEST 8: Multi-Panel Heatmaps
# ============================================================================

def test8_heatmaps(
    device: torch.device,
    high_dab_tiles: list[str],
) -> dict:
    """Generate 3x2 heatmap panels for 5 blue + 5 DAB tiles.

    Panel layout:
      A: Original RGB tile
      B: DAB channel (stain deconvolution)
      C: Model seed confidence map (raw FCN output)
      D: Model boundary probability map (raw FCN output)
      E: Bar-filter response overlaid on RGB
      F: Final cell boundaries overlaid on RGB
    """
    logger.info("=" * 70)
    logger.info("TEST 8: Multi-Panel Heatmaps")
    logger.info("=" * 70)

    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    models = load_models(device, need_our=True, need_baseline=False)
    target_cn = torch.tensor([1, 1], dtype=torch.long, device=device)

    # Select 5 blue + 5 DAB tiles
    tiles_to_process = [(p, "blue") for p in BLUE_TILES[:5]] + [
        (p, "dab") for p in high_dab_tiles[:5]
    ]

    per_tile: list[dict] = []

    for rel_path, cat in tqdm(tiles_to_process, desc="Test 8: Heatmaps"):
        img = load_tile(rel_path)
        img_t = img_to_tensor(img, device)

        # A: Original RGB (already have img)

        # B: DAB channel
        dab = extract_dab(img)

        # C & D: Raw FCN output (seed confidence + boundary map)
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            raw_output = models["our"].fcn(img_t)
        # raw_output shape: (1, 10, H, W)
        # Channels: [5 nuclei outputs, 5 cell outputs]
        # Seed channel: index 4 (nuclei), 9 (cells)
        # Boundary channels: indices 0-3 (nuclei), 5-8 (cells)
        raw_np = raw_output[0].cpu().float().numpy()  # (10, H, W)

        # Cell seed confidence (channel 9)
        cell_seed = raw_np[9]
        # Cell boundary probability (mean of channels 5-8)
        cell_boundary = np.mean(raw_np[5:9], axis=0)

        # E: Bar-filter response
        bar_response = apply_bar_filters(dab, device)

        # F: Final cell boundaries
        _, cell_labels = run_our_model(models["our"], img_t, target_cn)
        n_cells = int((np.unique(cell_labels) > 0).sum())

        del img_t
        torch.cuda.empty_cache()

        # ---- Build 3x2 figure ----
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Panel A: Original RGB
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("A: Original RGB", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")

        # Panel B: DAB channel
        im_b = axes[0, 1].imshow(dab, cmap="RdBu_r", vmin=0, vmax=0.5)
        axes[0, 1].set_title("B: DAB Channel (deconvolved)", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")
        plt.colorbar(im_b, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Panel C: Seed confidence map
        im_c = axes[0, 2].imshow(cell_seed, cmap="hot", vmin=np.percentile(cell_seed, 5))
        axes[0, 2].set_title(
            "C: Cell Seed Confidence (raw FCN ch9)", fontsize=12, fontweight="bold"
        )
        axes[0, 2].axis("off")
        plt.colorbar(im_c, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # Panel D: Boundary probability map
        im_d = axes[1, 0].imshow(cell_boundary, cmap="magma")
        axes[1, 0].set_title(
            "D: Cell Boundary Map (raw FCN ch5-8 mean)", fontsize=12, fontweight="bold"
        )
        axes[1, 0].axis("off")
        plt.colorbar(im_d, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # Panel E: Bar-filter response overlaid on RGB
        overlay_bar = img.copy().astype(np.float32)
        bar_norm = np.clip(bar_response / (np.percentile(bar_response, 99) + 1e-6), 0, 1)
        # Overlay as green channel boost
        overlay_bar[:, :, 1] = np.clip(
            overlay_bar[:, :, 1] + bar_norm * 200, 0, 255
        )
        axes[1, 1].imshow(overlay_bar.astype(np.uint8))
        axes[1, 1].set_title(
            "E: Bar-Filter Response (green overlay)", fontsize=12, fontweight="bold"
        )
        axes[1, 1].axis("off")

        # Panel F: Final cell boundaries
        overlay_cells = img.copy()
        draw_contours(overlay_cells, cell_labels, color=(0, 255, 0), thickness=2)
        axes[1, 2].imshow(overlay_cells)
        axes[1, 2].set_title(
            f"F: Final Cell Boundaries (n={n_cells})", fontsize=12, fontweight="bold"
        )
        axes[1, 2].axis("off")

        # Title
        slide_name = Path(rel_path).parent.name
        tile_name = Path(rel_path).stem
        fig.suptitle(
            f"{cat.upper()} | {slide_name}/{tile_name} | DAB mean={np.mean(dab):.4f} | "
            f"Cells={n_cells}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = f"heatmap_{cat}_{slide_name}_{tile_name}.png"
        fig.savefig(str(HEATMAP_DIR / fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

        per_tile.append(
            {
                "tile": rel_path,
                "category": cat,
                "cells": n_cells,
                "dab_mean": round(float(np.mean(dab)), 4),
                "saved_as": fname,
            }
        )

    summary = {
        "heatmap_dir": str(HEATMAP_DIR),
        "n_heatmaps": len(per_tile),
        "panels": [
            "A: Original RGB",
            "B: DAB channel (stain deconvolution)",
            "C: Cell seed confidence map (raw FCN channel 9)",
            "D: Cell boundary probability map (raw FCN channels 5-8 mean)",
            "E: Bar-filter response overlaid on RGB",
            "F: Final cell boundaries overlaid on RGB",
        ],
        "per_tile": per_tile,
    }

    logger.info("  Saved %d heatmaps to %s", len(per_tile), HEATMAP_DIR)

    del models
    torch.cuda.empty_cache()
    return summary


# ============================================================================
# Main orchestrator
# ============================================================================

def print_summary_table(results: dict) -> None:
    """Print a comprehensive summary table to stdout."""
    print()
    print("=" * 80)
    print("  COMPREHENSIVE SELECTIVITY VALIDATION -- SUMMARY")
    print("=" * 80)
    print()

    # Test 1
    if "test1" in results:
        t = results["test1"]
        print("TEST 1: Visual Overlays on Blue vs DAB-Positive Tiles")
        print(f"  Blue tiles:  our model = {t.get('blue_our_mean', 0):.1f} cells/tile,"
              f"  baseline = {t.get('blue_baseline_mean', 0):.1f} nuclei/tile")
        print(f"  DAB tiles:   our model = {t.get('dab_our_mean', 0):.1f} cells/tile,"
              f"  baseline = {t.get('dab_baseline_mean', 0):.1f} nuclei/tile")
        print(f"  Overlays saved to: {t.get('overlay_dir', 'N/A')}")
        print()

    # Test 2
    if "test2" in results:
        t = results["test2"]
        if "error" not in t:
            print("TEST 2: DAB Measurement at Detected Boundaries")
            print(f"  Negative tissue: {t.get('negative_cells_measured', 0)} cells,"
                  f"  mean membrane DAB = {t.get('neg_membrane_dab_mean', 0):.4f}")
            print(f"  Positive tissue: {t.get('positive_cells_measured', 0)} cells,"
                  f"  mean membrane DAB = {t.get('pos_membrane_dab_mean', 0):.4f}")
            print(f"  Ratio (neg/pos): {t.get('neg_vs_pos_dab_ratio', 'N/A')}")
            print(f"  {t.get('interpretation', '')}")
        print()

    # Test 3
    if "test3" in results:
        t = results["test3"]
        if "error" not in t:
            print("TEST 3: H&E Slide Test")
            print(f"  H&E tiles tested: {t.get('n_tiles', 0)}")
            print(f"  Our model: {t.get('our_mean_cells', 0):.1f} cells/tile"
                  f"  (zero-detection tiles: {t.get('our_zero_tiles', 0)})")
            print(f"  Baseline:  {t.get('baseline_mean_nuclei', 0):.1f} nuclei/tile")
            print(f"  Ratio:     {t.get('ratio', 'N/A')}")
            print(f"  Verdict:   {t.get('verdict', 'N/A')}")
        else:
            print(f"TEST 3: SKIPPED -- {t['error']}")
        print()

    # Test 4
    if "test4" in results:
        t = results["test4"]
        print("TEST 4: Cross-Reference with Baseline")
        print(f"  Our cells on blue tiles: {t.get('total_our_cells', 0)}")
        print(f"  Matched to baseline nuclei (<10px): {t.get('total_matched', 0)}"
              f"  ({t.get('overall_match_pct', 0):.1f}%)")
        print(f"  {t.get('interpretation', '')}")
        print()

    # Test 5
    if "test5" in results:
        t = results["test5"]
        if "error" not in t:
            print("TEST 5: Virchow2 Epithelial Classification")
            print(f"  Blue tiles in epithelial cluster: "
                  f"{t.get('blue_tiles_in_epithelial_cluster', 0)} / "
                  f"{t.get('blue_tiles_total', 0)}"
                  f"  ({t.get('blue_epi_tile_pct', 0):.1f}%)")
            print(f"  Blue cells in epithelial cluster: "
                  f"{t.get('blue_cells_in_epi', 0)} / "
                  f"{t.get('blue_cells_total', 0)}"
                  f"  ({t.get('blue_cells_epi_pct', 0):.1f}%)")
        else:
            print(f"TEST 5: SKIPPED -- {t['error']}")
        print()

    # Test 6
    if "test6" in results:
        t = results["test6"]
        print("TEST 6: Fluorescence Model Comparison")
        if "blue" in t:
            print(f"  Blue tiles ({t.get('blue_tiles', 0)}):")
            print(f"    Our model:    {t['blue']['our']['mean']:.1f} cells/tile")
            print(f"    Fluorescence: {t['blue']['fluoro']['mean']:.1f} cells/tile")
            print(f"    Baseline:     {t['blue']['baseline']['mean']:.1f} nuclei/tile")
        if "dab" in t:
            print(f"  DAB tiles ({t.get('dab_tiles', 0)}):")
            print(f"    Our model:    {t['dab']['our']['mean']:.1f} cells/tile")
            print(f"    Fluorescence: {t['dab']['fluoro']['mean']:.1f} cells/tile")
            print(f"    Baseline:     {t['dab']['baseline']['mean']:.1f} nuclei/tile")
        print(f"  {t.get('interpretation', '')}")
        print()

    # Test 7
    if "test7" in results:
        t = results["test7"]
        print("TEST 7: Training Mask Overlap")
        print(f"  Blue tiles with GT nuclei: {t.get('tiles_with_gt_nuclei', 0)} / "
              f"{t.get('n_tiles', 0)}")
        print(f"  Blue tiles with GT cells:  {t.get('tiles_with_gt_cells', 0)} / "
              f"{t.get('n_tiles', 0)}")
        print(f"  Blue tiles with our detections: "
              f"{t.get('tiles_with_our_detections', 0)} / {t.get('n_tiles', 0)}")
        print(f"  GT-zero but our-nonzero: {t.get('tiles_gt_zero_our_nonzero', 0)}")
        print(f"  {t.get('interpretation', '')}")
        print()

    # Test 8
    if "test8" in results:
        t = results["test8"]
        print("TEST 8: Multi-Panel Heatmaps")
        print(f"  Generated: {t.get('n_heatmaps', 0)} heatmap panels")
        print(f"  Saved to:  {t.get('heatmap_dir', 'N/A')}")
        print()

    # Overall verdict
    print("=" * 80)
    print("  OVERALL SELECTIVITY ASSESSMENT")
    print("=" * 80)

    verdicts = []

    # From Test 1: ratio of blue vs DAB cell counts
    if "test1" in results:
        t = results["test1"]
        blue_m = t.get("blue_our_mean", 0)
        dab_m = t.get("dab_our_mean", 1)
        if dab_m > 0:
            r = blue_m / dab_m
            if r < 0.10:
                verdicts.append(("Test 1", "STRONG selectivity", r))
            elif r < 0.40:
                verdicts.append(("Test 1", "MODERATE selectivity", r))
            else:
                verdicts.append(("Test 1", "WEAK selectivity", r))

    # From Test 3: H&E verdict
    if "test3" in results and "verdict" in results["test3"]:
        verdicts.append(("Test 3", results["test3"]["verdict"], results["test3"].get("ratio", 0)))

    # From Test 4: match percentage
    if "test4" in results:
        mp = results["test4"].get("overall_match_pct", 0)
        if mp > 80:
            verdicts.append(("Test 4", "Detections are REAL cells", mp))
        elif mp > 50:
            verdicts.append(("Test 4", "Mostly real cells", mp))
        else:
            verdicts.append(("Test 4", "Possibly artifacts", mp))

    for test_name, verdict_str, metric in verdicts:
        print(f"  {test_name}: {verdict_str} (metric={metric:.4f})")

    print()
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive selectivity validation for brightfield_cells_nuclei.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Comma-separated test numbers to run (e.g., '1,2,4'). Default: none.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 8 tests.",
    )
    parser.add_argument(
        "--he-tiles",
        type=int,
        default=50,
        help="Number of H&E tiles to extract for test 3 (default: 50).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "VRAM: %.1f GB total, %.1f GB free",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
            (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9,
        )

    # Determine which tests to run
    if args.all:
        tests_to_run = {1, 2, 3, 4, 5, 6, 7, 8}
    elif args.test:
        tests_to_run = {int(t.strip()) for t in args.test.split(",")}
    else:
        logger.error("No tests specified. Use --all or --test 1,2,3,...")
        return

    logger.info("Tests to run: %s", sorted(tests_to_run))
    print()

    # Create output directories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-scan: find high-DAB tiles (needed by tests 1, 5, 6, 8)
    need_high_dab = bool(tests_to_run & {1, 5, 6, 8})
    if need_high_dab:
        logger.info("Pre-scanning for high-DAB tiles ...")
        high_dab_tiles = find_high_dab_tiles(n=10)
        logger.info("  Found %d high-DAB tiles", len(high_dab_tiles))
        for i, t in enumerate(high_dab_tiles):
            logger.info("    %d: %s", i + 1, t)
    else:
        high_dab_tiles = []

    # Load any existing partial results
    all_results: dict[str, Any] = {}
    if RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON) as f:
                all_results = json.load(f)
            logger.info("Loaded existing results from %s", RESULTS_JSON)
        except Exception:
            pass

    t0 = time.time()

    # ---- Run selected tests ----

    if 1 in tests_to_run:
        try:
            all_results["test1"] = test1_visual_overlays(device, high_dab_tiles)
        except Exception as exc:
            logger.exception("Test 1 failed: %s", exc)
            all_results["test1"] = {"error": str(exc)}

    if 2 in tests_to_run:
        try:
            all_results["test2"] = test2_boundary_dab_measurement(device)
        except Exception as exc:
            logger.exception("Test 2 failed: %s", exc)
            all_results["test2"] = {"error": str(exc)}

    if 3 in tests_to_run:
        try:
            all_results["test3"] = test3_he_slide(device, n_tiles=args.he_tiles)
        except Exception as exc:
            logger.exception("Test 3 failed: %s", exc)
            all_results["test3"] = {"error": str(exc)}

    if 4 in tests_to_run:
        try:
            all_results["test4"] = test4_cross_reference(device)
        except Exception as exc:
            logger.exception("Test 4 failed: %s", exc)
            all_results["test4"] = {"error": str(exc)}

    if 5 in tests_to_run:
        try:
            all_results["test5"] = test5_virchow2_epithelial(device, high_dab_tiles)
        except Exception as exc:
            logger.exception("Test 5 failed: %s", exc)
            all_results["test5"] = {"error": str(exc)}

    if 6 in tests_to_run:
        try:
            all_results["test6"] = test6_fluorescence_comparison(device, high_dab_tiles)
        except Exception as exc:
            logger.exception("Test 6 failed: %s", exc)
            all_results["test6"] = {"error": str(exc)}

    if 7 in tests_to_run:
        try:
            all_results["test7"] = test7_training_mask_overlap(device)
        except Exception as exc:
            logger.exception("Test 7 failed: %s", exc)
            all_results["test7"] = {"error": str(exc)}

    if 8 in tests_to_run:
        try:
            all_results["test8"] = test8_heatmaps(device, high_dab_tiles)
        except Exception as exc:
            logger.exception("Test 8 failed: %s", exc)
            all_results["test8"] = {"error": str(exc)}

    elapsed = time.time() - t0

    # ---- Save results ----
    all_results["metadata"] = {
        "tests_run": sorted(tests_to_run),
        "elapsed_seconds": round(elapsed, 1),
        "device": str(device),
        "blue_tiles": BLUE_TILES,
        "high_dab_tiles": high_dab_tiles,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", RESULTS_JSON)

    # ---- Print summary ----
    print_summary_table(all_results)

    logger.info("Total elapsed: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
