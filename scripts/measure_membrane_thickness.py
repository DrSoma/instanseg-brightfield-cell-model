#!/usr/bin/env python3
"""Per-cell membrane thickness (FWHM) measurement + CLDN18.2 composite scoring.

Adds membrane thickness measurement to the pipeline Parquet output via
radial DAB profiling (FWHM of perpendicular cross-sections), then computes
a composite CLDN18.2 grade that mirrors VENTANA (43-14A) pathologist scoring:
  - Grade 0 (Negative): No membrane staining
  - Grade 1+ (Weak):    Light brown, partial membrane, normal thickness
  - Grade 2+ (Moderate): Chocolate brown, partial/circumferential, normal thickness
  - Grade 3+ (Strong):  Dark brown/black, circumferential, THICKENED membrane

New Parquet columns:
  - membrane_thickness_px  (float64): median FWHM in pixels (level 1)
  - membrane_thickness_um  (float64): median FWHM in microns
  - cldn18_composite_grade (int8):    0/1/2/3 VENTANA-style grade

Outputs:
  - Updated Parquet files at /tmp/pipeline_comparison/v2_cells_nuclei/cell_data/
  - Composite heatmaps (PNG) at ~/Downloads/
  - Thickness heatmaps (PNG) at ~/Downloads/
  - Summary JSON at evaluation/membrane_thickness_results.json

Usage:
    /home/fernandosoto/claudin18_venv/bin/python \
        scripts/measure_membrane_thickness.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Headless rendering
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import openslide
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter
from shapely import from_wkb
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLIDES_DIR = Path("/tmp/bc_slides")
PARQUET_DIR = Path("/tmp/pipeline_comparison/v2_cells_nuclei/cell_data")
OUTPUT_DIR = Path("/home/fernandosoto/Downloads")
EVAL_DIR = PROJECT_ROOT / "evaluation"

SLIDES = {
    "BC_ClassII": {
        "slide": SLIDES_DIR / "BC_ClassII.ndpi",
        "parquet": PARQUET_DIR / "BC_ClassII_cells.parquet",
    },
    "BC_ClassIII": {
        "slide": SLIDES_DIR / "BC_ClassIII.ndpi",
        "parquet": PARQUET_DIR / "BC_ClassIII_cells.parquet",
    },
}

# Stain deconvolution vectors (calibrated for CLDN18 slides)
STAIN_H = np.array([0.786, 0.593, 0.174])
STAIN_DAB = np.array([0.215, 0.422, 0.881])
STAIN_R = np.array([0.547, -0.799, 0.249])

# FWHM measurement parameters
N_RAYS = 16              # Number of perpendicular rays per cell
RAY_EXTENT_PX = 20       # Pixels inward+outward from boundary (at level 1)
MIN_DAB_SIGNAL = 0.05    # Minimum DAB OD to consider a peak
PIXEL_SIZE_UM = 0.4598   # Level 1 MPP for these NDPI slides

# Composite scoring thresholds (PRELIMINARY — need pathologist calibration)
THICKNESS_THRESHOLD_PX = 8    # Normal vs thickened (at level 1, ~3.7 um)
COMPLETENESS_THRESHOLD = 0.5  # Partial vs circumferential
# DAB intensity: raw OD thresholds (not bar-filter weighted)
DAB_NEGATIVE_THRESHOLD = 0.05
DAB_WEAK_THRESHOLD = 0.15
DAB_STRONG_THRESHOLD = 0.30

# Tile grouping for efficient WSI reads
TILE_SIZE_L0 = 1024     # Group cells into tiles of this size at level 0
TILE_PADDING_L0 = 80    # Padding around tiles

# Heatmap constants
HEATMAP_WIDTH = 8192
GAUSSIAN_SIGMA = 3.0
GAP_FILL_SIZE = 12
OVERLAY_ALPHA = 0.55
TISSUE_THRESHOLD = 220
PERCENTILE_LO = 2
PERCENTILE_HI = 98
MIN_DISPLAY_THRESHOLD = 0.005


# ── Stain Deconvolution ─────────────────────────────────────────────────────


def _build_stain_inverse() -> np.ndarray:
    """Build inverse stain matrix for RGB -> OD deconvolution."""
    matrix = np.array([STAIN_H, STAIN_DAB, STAIN_R], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms
    return np.linalg.inv(matrix)


def extract_dab(tile_rgb: np.ndarray, inv_matrix: np.ndarray) -> np.ndarray:
    """Extract DAB optical density from an RGB tile.

    Returns (H, W) float32 DAB OD, values >= 0.
    """
    od = -np.log(np.clip(tile_rgb.astype(np.float64) / 255.0, 1 / 255, 1))
    return np.clip((od @ inv_matrix)[:, :, 1], 0, None).astype(np.float32)


# ── FWHM Measurement ────────────────────────────────────────────────────────


def measure_membrane_fwhm(
    dab_tile: np.ndarray,
    cell_polygon,
    tile_origin_l0: tuple[int, int],
    level_ds: float,
    n_rays: int = N_RAYS,
    ray_extent: int = RAY_EXTENT_PX,
) -> float:
    """Measure membrane thickness via radial DAB profiles (FWHM).

    For each of n_rays evenly-spaced points on the cell boundary:
    1. Compute the outward normal direction
    2. Cast a ray perpendicular to the boundary (inward+outward)
    3. Sample DAB values along the ray
    4. Find the FWHM of the DAB peak

    Args:
        dab_tile: (H, W) float32 DAB optical density at read level.
        cell_polygon: Shapely Polygon in level-0 coordinates.
        tile_origin_l0: (x0, y0) of tile top-left in level-0 coords.
        level_ds: Downsample factor of the read level.
        n_rays: Number of rays to cast around the boundary.
        ray_extent: Pixels to extend in each direction from boundary.

    Returns:
        Median FWHM in pixels at the read level, or NaN if unmeasurable.
    """
    boundary = cell_polygon.boundary
    if boundary.is_empty or boundary.length < 4:
        return np.nan

    tile_h, tile_w = dab_tile.shape
    x0_l0, y0_l0 = tile_origin_l0
    fwhms = []

    for i in range(n_rays):
        frac = i / n_rays

        # Sample point on boundary
        pt = boundary.interpolate(frac, normalized=True)

        # Estimate tangent from nearby points
        eps = 0.01
        frac_before = max(0.0, frac - eps)
        frac_after = min(1.0, frac + eps)
        pt_before = boundary.interpolate(frac_before, normalized=True)
        pt_after = boundary.interpolate(frac_after, normalized=True)

        dx = pt_after.x - pt_before.x
        dy = pt_after.y - pt_before.y
        tangent_norm = np.sqrt(dx * dx + dy * dy) + 1e-8

        # Normal = perpendicular to tangent (inward-pointing)
        nx = -dy / tangent_norm
        ny = dx / tangent_norm

        # Convert boundary point to tile-local pixel coordinates at read level
        pt_lx = (pt.x - x0_l0) / level_ds
        pt_ly = (pt.y - y0_l0) / level_ds

        # Sample DAB along normal ray: -ray_extent to +ray_extent pixels
        profile = np.zeros(2 * ray_extent + 1, dtype=np.float32)
        for j, d in enumerate(range(-ray_extent, ray_extent + 1)):
            sx = int(round(pt_lx + d * nx))
            sy = int(round(pt_ly + d * ny))
            if 0 <= sx < tile_w and 0 <= sy < tile_h:
                profile[j] = dab_tile[sy, sx]
            # else remains 0

        # Find FWHM of the DAB peak
        peak_val = profile.max()
        if peak_val < MIN_DAB_SIGNAL:
            continue

        half_max = peak_val / 2.0
        above_half = profile >= half_max

        # Find contiguous runs above half-maximum
        padded = np.concatenate([[False], above_half, [False]])
        runs = np.diff(padded.astype(np.int8))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]

        if len(starts) == 0 or len(ends) == 0:
            continue

        # Take the longest contiguous run (handles multiple peaks)
        widths = ends - starts
        fwhm = int(widths.max())
        if fwhm >= 1:
            fwhms.append(fwhm)

    if len(fwhms) >= 3:  # Need at least 3 valid rays
        return float(np.median(fwhms))
    return np.nan


# ── Composite CLDN18.2 Grade ────────────────────────────────────────────────


def compute_cldn18_grade(
    thickness_px: float,
    completeness: float,
    dab_intensity: float,
) -> int:
    """Assign VENTANA-style CLDN18.2 grade from three features.

    Thresholds are PRELIMINARY and require pathologist calibration.
    Based on measured FWHM: population median ~10px, range 1-20px at level 1.

    VENTANA (43-14A) criteria:
      0  Negative:  No membrane DAB
      1+ Weak:      Light brown, partial membrane, normal thickness
      2+ Moderate:  Brown, partial/circumferential, normal thickness
      3+ Strong:    Dark brown, circumferential, THICKENED membrane

    Args:
        thickness_px: Median FWHM in pixels at level 1 (~0.46 um/px).
        completeness: Fraction of boundary with positive bar-filter response.
        dab_intensity: Raw DAB optical density at membrane.

    Returns:
        Grade 0, 1, 2, or 3.
    """
    # Handle missing values
    if np.isnan(thickness_px) or np.isnan(completeness) or np.isnan(dab_intensity):
        return 0

    # Feature classification
    is_thickened = thickness_px >= THICKNESS_THRESHOLD_PX
    is_circumferential = completeness >= COMPLETENESS_THRESHOLD

    # Intensity gating
    if dab_intensity < DAB_NEGATIVE_THRESHOLD:
        return 0  # Negative: no staining

    if dab_intensity < DAB_WEAK_THRESHOLD:
        return 1  # Weak regardless of other features

    # Moderate-to-strong range (DAB >= 0.15)
    if is_thickened:
        return 3  # Strong: thickened membrane is the hallmark of 3+

    if is_circumferential and dab_intensity >= DAB_STRONG_THRESHOLD:
        return 3  # Strong: circumferential + high intensity

    if is_circumferential:
        return 2  # Moderate: circumferential, normal thickness

    # Partial membrane, normal thickness, moderate intensity
    if dab_intensity >= DAB_STRONG_THRESHOLD:
        return 2  # Moderate: partial but intense

    return 1  # Weak: partial, normal thickness, moderate intensity


def compute_hscore(grades: np.ndarray) -> float:
    """H-score from composite grades: 1*%1+ + 2*%2+ + 3*%3+. Range 0-300."""
    if len(grades) == 0:
        return 0.0
    n = len(grades)
    return float(
        1 * (grades == 1).sum() / n * 100
        + 2 * (grades == 2).sum() / n * 100
        + 3 * (grades == 3).sum() / n * 100
    )


# ── Step 1: Measure Thickness per Cell ──────────────────────────────────────


def measure_thickness_for_slide(
    slide_id: str,
    slide_path: Path,
    parquet_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure membrane FWHM for every cell in a slide's Parquet file.

    Groups cells by tile for efficient WSI reading. Reads each tile once
    and processes all cells within it.

    Returns:
        (thickness_px, raw_dab_at_membrane) arrays of shape (n_cells,).
    """
    table = pq.read_table(parquet_path)
    n_cells = table.num_rows

    polygon_wkb_col = table.column("polygon_wkb").to_pylist()
    cx_col = table.column("centroid_x").to_numpy()
    cy_col = table.column("centroid_y").to_numpy()

    thickness_px = np.full(n_cells, np.nan, dtype=np.float64)
    raw_membrane_dab = np.full(n_cells, np.nan, dtype=np.float64)

    stain_inv = _build_stain_inverse()

    slide = openslide.OpenSlide(str(slide_path))
    try:
        # Use level 1: ~0.46 um/px
        best_level = 1
        level_ds = slide.level_downsamples[best_level]
        slide_w, slide_h = slide.dimensions

        # Group cells into spatial tile bins
        bins: dict[tuple[int, int], list[int]] = {}
        for i in range(n_cells):
            gx = int(cx_col[i] / TILE_SIZE_L0)
            gy = int(cy_col[i] / TILE_SIZE_L0)
            key = (gx, gy)
            if key not in bins:
                bins[key] = []
            bins[key].append(i)

        logger.info("  %s: %d cells in %d tiles", slide_id, n_cells, len(bins))

        for (gx, gy), cell_indices in tqdm(
            bins.items(),
            desc=f"  {slide_id} thickness",
            unit="tile",
        ):
            # Tile region in level-0 coords
            x0_l0 = max(0, int(gx * TILE_SIZE_L0 - TILE_PADDING_L0))
            y0_l0 = max(0, int(gy * TILE_SIZE_L0 - TILE_PADDING_L0))
            read_w_l0 = min(
                int(TILE_SIZE_L0 + 2 * TILE_PADDING_L0),
                slide_w - x0_l0,
            )
            read_h_l0 = min(
                int(TILE_SIZE_L0 + 2 * TILE_PADDING_L0),
                slide_h - y0_l0,
            )

            if read_w_l0 <= 0 or read_h_l0 <= 0:
                continue

            # Read at level 1
            read_w_lv = max(1, int(read_w_l0 / level_ds))
            read_h_lv = max(1, int(read_h_l0 / level_ds))

            region_img = slide.read_region(
                (x0_l0, y0_l0), best_level, (read_w_lv, read_h_lv)
            )
            tile_rgb = np.array(region_img.convert("RGB"))
            dab = extract_dab(tile_rgb, stain_inv)

            for cell_idx in cell_indices:
                wkb_data = polygon_wkb_col[cell_idx]
                if wkb_data is None:
                    continue

                try:
                    poly = from_wkb(wkb_data)
                except Exception:
                    continue

                if poly.is_empty:
                    continue

                # Measure FWHM
                fwhm = measure_membrane_fwhm(
                    dab, poly, (x0_l0, y0_l0), level_ds,
                )
                thickness_px[cell_idx] = fwhm

                # Also measure raw DAB at the boundary for composite scoring
                # (unweighted mean DAB in a thin ring around the boundary)
                bx0, by0, bx1, by1 = poly.bounds
                lx0 = max(0, int((bx0 - x0_l0) / level_ds) - 2)
                ly0 = max(0, int((by0 - y0_l0) / level_ds) - 2)
                lx1 = min(read_w_lv, int((bx1 - x0_l0) / level_ds) + 3)
                ly1 = min(read_h_lv, int((by1 - y0_l0) / level_ds) + 3)

                if lx1 <= lx0 or ly1 <= ly0:
                    continue

                # Rasterize polygon to mask
                ext_coords = np.array(poly.exterior.coords)
                local_coords = (
                    ext_coords - np.array([x0_l0, y0_l0])
                ) / level_ds
                local_pts = local_coords.astype(np.int32)

                cell_mask = np.zeros(
                    (read_h_lv, read_w_lv), dtype=np.uint8
                )
                cv2.fillPoly(cell_mask, [local_pts], 1)

                # Thin boundary ring (3px erosion)
                eroded = cv2.erode(
                    cell_mask, np.ones((3, 3), np.uint8), iterations=1
                )
                ring = (cell_mask > 0) & (eroded == 0)

                ring_crop = ring[ly0:ly1, lx0:lx1]
                dab_crop = dab[ly0:ly1, lx0:lx1]

                if ring_crop.sum() >= 3:
                    raw_membrane_dab[cell_idx] = float(
                        dab_crop[ring_crop].mean()
                    )

    finally:
        slide.close()

    return thickness_px, raw_membrane_dab


# ── Step 2: Update Parquet with New Columns ─────────────────────────────────


def update_parquet(
    parquet_path: Path,
    thickness_px: np.ndarray,
    raw_membrane_dab: np.ndarray,
    completeness: np.ndarray,
) -> pa.Table:
    """Add thickness and composite grade columns to the Parquet file.

    Returns the updated table (also written to disk).
    """
    table = pq.read_table(parquet_path)
    n = table.num_rows

    # Compute derived columns
    thickness_um = thickness_px * PIXEL_SIZE_UM

    # Composite grades
    grades = np.zeros(n, dtype=np.int8)
    for i in range(n):
        grades[i] = compute_cldn18_grade(
            thickness_px[i],
            completeness[i],
            raw_membrane_dab[i],
        )

    # Remove old columns if they exist (in case of re-run)
    new_cols = [
        "membrane_thickness_px",
        "membrane_thickness_um",
        "raw_membrane_dab",
        "cldn18_composite_grade",
    ]
    existing = table.column_names
    for col_name in new_cols:
        if col_name in existing:
            idx = existing.index(col_name)
            table = table.remove_column(idx)
            existing = table.column_names

    # Add new columns
    table = table.append_column(
        "membrane_thickness_px",
        pa.array(thickness_px, type=pa.float64()),
    )
    table = table.append_column(
        "membrane_thickness_um",
        pa.array(thickness_um, type=pa.float64()),
    )
    table = table.append_column(
        "raw_membrane_dab",
        pa.array(raw_membrane_dab, type=pa.float64()),
    )
    table = table.append_column(
        "cldn18_composite_grade",
        pa.array(grades, type=pa.int8()),
    )

    pq.write_table(table, parquet_path, compression="zstd")
    return table


# ── Step 3: Heatmap Generation ──────────────────────────────────────────────


def scatter_to_grid(
    cx: np.ndarray,
    cy: np.ndarray,
    values: np.ndarray,
    scale: float,
    width: int,
    height: int,
    mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Map cell coordinates to a thumbnail-resolution grid."""
    heatmap = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.float64)

    px = (cx * scale).astype(np.int64)
    py = (cy * scale).astype(np.int64)

    r = 4  # cell dot radius in heatmap pixels
    mask = (px >= r) & (px < width - r) & (py >= r) & (py < height - r)
    px = px[mask]
    py = py[mask]
    val = values[mask]

    nan_mask = np.isnan(val)
    val_clean = np.where(nan_mask, 0.0, val)
    valid_flags = (~nan_mask).astype(np.float64)

    # Draw each cell as a filled circle for visibility
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    circle = (xx**2 + yy**2) <= r**2
    dy = yy[circle]
    dx = xx[circle]
    for i in range(len(px)):
        ys = py[i] + dy
        xs = px[i] + dx
        heatmap[ys, xs] += val_clean[i]
        counts[ys, xs] += valid_flags[i]

    if mode == "mean":
        valid = counts > 0
        heatmap[valid] /= counts[valid]

    return heatmap.astype(np.float32), counts.astype(np.float32)


def build_heatmap(
    raw: np.ndarray,
    tissue_mask: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Normalize, gap-fill, smooth, and tissue-mask a raw heatmap."""
    valid = counts > 0
    if valid.sum() == 0:
        return np.zeros_like(raw)

    vals = raw[valid]
    lo = np.percentile(vals, PERCENTILE_LO)
    hi = np.percentile(vals, PERCENTILE_HI)
    denom = hi - lo
    if denom < 1e-8:
        denom = 1.0
    heatmap_norm = np.clip((raw - lo) / denom, 0.0, 1.0)

    heatmap_filled = maximum_filter(heatmap_norm, size=GAP_FILL_SIZE)
    heatmap_smooth = gaussian_filter(heatmap_filled, sigma=GAUSSIAN_SIGMA)
    heatmap_smooth[~tissue_mask] = 0.0

    return heatmap_smooth


def make_composite_heatmap(
    slide_path: Path,
    table: pa.Table,
    slide_id: str,
) -> list[Path]:
    """Generate composite grade + thickness heatmaps for one slide.

    Returns list of output file paths.
    """
    outputs: list[Path] = []

    # Load thumbnail
    slide = openslide.OpenSlide(str(slide_path))
    w0, h0 = slide.dimensions
    target_h = int(h0 * (HEATMAP_WIDTH / w0))
    thumb = slide.get_thumbnail((HEATMAP_WIDTH, target_h))
    thumb_np = np.array(thumb.convert("RGB"))
    slide.close()

    actual_h, actual_w = thumb_np.shape[:2]
    scale = actual_w / w0

    # Tissue mask
    gray = np.mean(thumb_np[:, :, :3], axis=2)
    tissue_mask = gray < TISSUE_THRESHOLD

    # Cell data
    cx = table.column("centroid_x").to_numpy()
    cy = table.column("centroid_y").to_numpy()
    grades = table.column("cldn18_composite_grade").to_numpy().astype(np.float32)
    thickness = table.column("membrane_thickness_px").to_numpy().astype(np.float32)

    # ── Composite Grade Heatmap ──
    # Use a discrete colormap: 0=gray, 1+=blue, 2+=orange, 3+=red
    grade_colors = {
        0: np.array([160, 160, 160, 140], dtype=np.uint8),  # gray
        1: np.array([65, 105, 225, 180], dtype=np.uint8),    # royal blue
        2: np.array([255, 165, 0, 200], dtype=np.uint8),     # orange
        3: np.array([220, 20, 60, 220], dtype=np.uint8),     # crimson red
    }

    # Scatter grades to grid (mode=mean gives fractional grades for blending)
    grade_raw, grade_counts = scatter_to_grid(
        cx, cy, grades, scale, actual_w, actual_h, mode="mean",
    )

    # Build RGBA overlay from grade values (vectorized)
    grade_rgba = np.zeros((actual_h, actual_w, 4), dtype=np.uint8)
    filled = maximum_filter(grade_raw, size=GAP_FILL_SIZE)
    smoothed = gaussian_filter(filled, sigma=GAUSSIAN_SIGMA)

    has_signal = grade_counts > 0
    has_signal_filled = (
        maximum_filter(has_signal.astype(np.float32), size=GAP_FILL_SIZE) > 0
    )

    active = tissue_mask & has_signal_filled
    rounded = np.clip(np.round(smoothed), 0, 3).astype(np.int32)
    for g, color in grade_colors.items():
        mask_g = active & (rounded == g)
        grade_rgba[mask_g] = color

    comp_path = OUTPUT_DIR / f"{slide_id}_composite_heatmap.png"
    Image.fromarray(grade_rgba, "RGBA").save(str(comp_path))
    outputs.append(comp_path)
    logger.info("  Saved composite heatmap: %s", comp_path)

    # Composite legend
    _make_grade_legend(OUTPUT_DIR / f"{slide_id}_composite_legend.png")
    outputs.append(OUTPUT_DIR / f"{slide_id}_composite_legend.png")

    # ── Thickness Heatmap (viridis) ──
    thick_raw, thick_counts = scatter_to_grid(
        cx, cy, thickness, scale, actual_w, actual_h, mode="mean",
    )
    thick_smooth = build_heatmap(thick_raw, tissue_mask, thick_counts)
    cmap = plt.get_cmap("viridis")
    thick_colored = (cmap(thick_smooth) * 255).astype(np.uint8)
    alpha_mask = tissue_mask & (thick_smooth > MIN_DISPLAY_THRESHOLD)
    thick_colored[:, :, 3] = np.where(
        alpha_mask, int(OVERLAY_ALPHA * 255), 0
    ).astype(np.uint8)

    thick_path = OUTPUT_DIR / f"{slide_id}_thickness_heatmap.png"
    Image.fromarray(thick_colored, "RGBA").save(str(thick_path))
    outputs.append(thick_path)
    logger.info("  Saved thickness heatmap: %s", thick_path)

    # Thickness colorbar
    vals = thick_raw[thick_counts > 0]
    if len(vals) > 0:
        lo = np.percentile(vals, PERCENTILE_LO)
        hi = np.percentile(vals, PERCENTILE_HI)
    else:
        lo, hi = 0, 20

    fig, ax = plt.subplots(figsize=(1.2, 4), dpi=120)
    norm = mcolors.Normalize(vmin=lo * PIXEL_SIZE_UM, vmax=hi * PIXEL_SIZE_UM)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax)
    cb.set_label("Membrane thickness (um)", fontsize=10)
    cb.ax.tick_params(labelsize=8)
    cbar_path = OUTPUT_DIR / f"{slide_id}_thickness_colorbar.png"
    fig.savefig(str(cbar_path), bbox_inches="tight", transparent=True, dpi=120)
    plt.close(fig)
    outputs.append(cbar_path)

    return outputs


def _make_grade_legend(out_path: Path) -> None:
    """Create a small legend PNG for composite grade colors."""
    fig, ax = plt.subplots(figsize=(2.5, 1.5), dpi=120)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 5)
    ax.set_axis_off()

    labels = ["0 (Negative)", "1+ (Weak)", "2+ (Moderate)", "3+ (Strong)"]
    colors = ["#A0A0A0", "#4169E1", "#FFA500", "#DC143C"]

    for i, (label, color) in enumerate(zip(labels, colors)):
        y = 4 - i
        ax.add_patch(plt.Rectangle((0.2, y - 0.35), 0.6, 0.7, color=color))
        ax.text(1.0, y, label, va="center", fontsize=9)

    ax.set_title("CLDN18.2 Composite Grade", fontsize=10, fontweight="bold")
    fig.savefig(str(out_path), bbox_inches="tight", transparent=True, dpi=120)
    plt.close(fig)


# ── Summary Statistics ───────────────────────────────────────────────────────


def compute_summary(
    slide_id: str,
    table: pa.Table,
) -> dict:
    """Compute summary statistics for one slide."""
    thickness_px = table.column("membrane_thickness_px").to_numpy()
    thickness_um = table.column("membrane_thickness_um").to_numpy()
    raw_dab = table.column("raw_membrane_dab").to_numpy()
    completeness = table.column("membrane_completeness").to_numpy()
    grades = table.column("cldn18_composite_grade").to_numpy()
    regions = table.column("region").to_pylist()

    n_total = len(grades)
    valid_thick = np.isfinite(thickness_px)

    grade_counts = Counter(int(g) for g in grades)
    composite_hscore = compute_hscore(grades)

    # Per-region stats
    region_stats = {}
    unique_regions = sorted(set(regions))
    for region in unique_regions:
        rmask = np.array([r == region for r in regions])
        rg = grades[rmask]
        rt = thickness_px[rmask]
        rv = np.isfinite(rt)

        rc = Counter(int(g) for g in rg)
        rhs = compute_hscore(rg)

        region_stats[region] = {
            "n_cells": int(rmask.sum()),
            "grade_distribution": {
                "0_neg": int(rc.get(0, 0)),
                "1_weak": int(rc.get(1, 0)),
                "2_moderate": int(rc.get(2, 0)),
                "3_strong": int(rc.get(3, 0)),
            },
            "composite_hscore": round(rhs, 1),
            "pct_positive_2plus": round(
                float((rg >= 2).sum() / max(len(rg), 1) * 100), 1
            ),
            "thickness_px": {
                "valid": int(rv.sum()),
                "mean": round(float(np.nanmean(rt[rv])), 2) if rv.sum() > 0 else None,
                "median": round(float(np.nanmedian(rt[rv])), 2) if rv.sum() > 0 else None,
                "std": round(float(np.nanstd(rt[rv])), 2) if rv.sum() > 0 else None,
            },
        }

    return {
        "slide_id": slide_id,
        "n_cells": n_total,
        "thickness_valid": int(valid_thick.sum()),
        "thickness_coverage_pct": round(
            100 * valid_thick.sum() / max(n_total, 1), 1
        ),
        "thickness_px_stats": {
            "mean": round(float(np.nanmean(thickness_px[valid_thick])), 2)
            if valid_thick.sum() > 0
            else None,
            "median": round(float(np.nanmedian(thickness_px[valid_thick])), 2)
            if valid_thick.sum() > 0
            else None,
            "std": round(float(np.nanstd(thickness_px[valid_thick])), 2)
            if valid_thick.sum() > 0
            else None,
            "p5": round(float(np.nanpercentile(thickness_px[valid_thick], 5)), 2)
            if valid_thick.sum() > 0
            else None,
            "p25": round(float(np.nanpercentile(thickness_px[valid_thick], 25)), 2)
            if valid_thick.sum() > 0
            else None,
            "p75": round(float(np.nanpercentile(thickness_px[valid_thick], 75)), 2)
            if valid_thick.sum() > 0
            else None,
            "p95": round(float(np.nanpercentile(thickness_px[valid_thick], 95)), 2)
            if valid_thick.sum() > 0
            else None,
        },
        "thickness_um_stats": {
            "mean": round(float(np.nanmean(thickness_um[valid_thick])), 2)
            if valid_thick.sum() > 0
            else None,
            "median": round(float(np.nanmedian(thickness_um[valid_thick])), 2)
            if valid_thick.sum() > 0
            else None,
        },
        "grade_distribution": {
            "0_neg": int(grade_counts.get(0, 0)),
            "1_weak": int(grade_counts.get(1, 0)),
            "2_moderate": int(grade_counts.get(2, 0)),
            "3_strong": int(grade_counts.get(3, 0)),
        },
        "composite_hscore": round(composite_hscore, 1),
        "pct_positive_2plus": round(
            float((grades >= 2).sum() / max(n_total, 1) * 100), 1
        ),
        "regions": region_stats,
    }


def print_summary_table(summaries: list[dict]) -> None:
    """Print formatted summary table for all slides."""
    print()
    print("=" * 95)
    print("MEMBRANE THICKNESS + COMPOSITE CLDN18.2 SCORE — RESULTS")
    print("=" * 95)

    for s in summaries:
        print(f"\n{'─' * 95}")
        print(f"  Slide: {s['slide_id']}")
        print(f"  Total cells: {s['n_cells']:,}")
        print(
            f"  Thickness measured: {s['thickness_valid']:,} / {s['n_cells']:,} "
            f"({s['thickness_coverage_pct']}%)"
        )
        print()

        # Thickness stats
        ts = s["thickness_px_stats"]
        tu = s["thickness_um_stats"]
        print("  MEMBRANE THICKNESS (at level 1, ~0.46 um/px):")
        print(
            f"    Pixels: mean={ts['mean']}, median={ts['median']}, "
            f"std={ts['std']}"
        )
        print(
            f"    IQR: [{ts['p25']}, {ts['p75']}] px  |  "
            f"5th-95th: [{ts['p5']}, {ts['p95']}] px"
        )
        print(
            f"    Microns: mean={tu['mean']} um, median={tu['median']} um"
        )
        print(
            f"    Thickened (>={THICKNESS_THRESHOLD_PX}px): "
            f"{s['thickness_px_stats'].get('mean', 0)} px mean"
        )
        print()

        # Grade distribution
        gd = s["grade_distribution"]
        n = s["n_cells"]
        print("  COMPOSITE CLDN18.2 GRADE (VENTANA-style):")
        print(
            f"    0 (Negative):  {gd['0_neg']:>7,} "
            f"({100 * gd['0_neg'] / max(n, 1):5.1f}%)"
        )
        print(
            f"    1+ (Weak):     {gd['1_weak']:>7,} "
            f"({100 * gd['1_weak'] / max(n, 1):5.1f}%)"
        )
        print(
            f"    2+ (Moderate): {gd['2_moderate']:>7,} "
            f"({100 * gd['2_moderate'] / max(n, 1):5.1f}%)"
        )
        print(
            f"    3+ (Strong):   {gd['3_strong']:>7,} "
            f"({100 * gd['3_strong'] / max(n, 1):5.1f}%)"
        )
        print(f"    Composite H-score: {s['composite_hscore']:.1f}")
        print(f"    CLDN18.2+ (>=2+):  {s['pct_positive_2plus']:.1f}%")
        print()

        # Per-region breakdown
        print("  PER-REGION BREAKDOWN:")
        header = (
            f"    {'Region':<20} {'Cells':>8} {'H-score':>9} "
            f"{'2+%':>7} {'Thick(px)':>10} {'0':>6} {'1+':>6} {'2+':>6} {'3+':>6}"
        )
        print(header)
        print(f"    {'─' * 85}")

        for region, rs in s["regions"].items():
            rgd = rs["grade_distribution"]
            t_med = rs["thickness_px"]["median"]
            t_str = f"{t_med:.1f}" if t_med is not None else "N/A"
            print(
                f"    {region:<20} {rs['n_cells']:>8,} "
                f"{rs['composite_hscore']:>9.1f} "
                f"{rs['pct_positive_2plus']:>6.1f}% "
                f"{t_str:>10} "
                f"{rgd['0_neg']:>6,} {rgd['1_weak']:>6,} "
                f"{rgd['2_moderate']:>6,} {rgd['3_strong']:>6,}"
            )

    print(f"\n{'=' * 95}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    t_start = time.time()

    print("=" * 70)
    print("  MEMBRANE THICKNESS MEASUREMENT + COMPOSITE CLDN18.2 SCORING")
    print("  VENTANA (43-14A) style: thickness + completeness + intensity")
    print("=" * 70)
    print()

    # Validate inputs
    for slide_id, cfg in SLIDES.items():
        if not cfg["slide"].exists():
            logger.error("Slide not found: %s", cfg["slide"])
            sys.exit(1)
        if not cfg["parquet"].exists():
            logger.error("Parquet not found: %s", cfg["parquet"])
            sys.exit(1)

    all_summaries: list[dict] = []
    all_heatmap_paths: list[Path] = []

    for slide_id, cfg in SLIDES.items():
        logger.info("=" * 60)
        logger.info("Processing: %s", slide_id)
        logger.info("=" * 60)

        # Step 1: Measure thickness
        logger.info("Step 1: Measuring membrane FWHM for %s ...", slide_id)
        t0 = time.time()
        thickness_px, raw_dab = measure_thickness_for_slide(
            slide_id, cfg["slide"], cfg["parquet"],
        )
        elapsed = time.time() - t0

        valid = np.isfinite(thickness_px)
        logger.info(
            "  Thickness measured: %d / %d cells (%.1f%%) in %.1fs",
            valid.sum(),
            len(thickness_px),
            100 * valid.sum() / max(len(thickness_px), 1),
            elapsed,
        )
        if valid.sum() > 0:
            logger.info(
                "  FWHM stats: mean=%.2f px, median=%.2f px, std=%.2f px",
                np.nanmean(thickness_px[valid]),
                np.nanmedian(thickness_px[valid]),
                np.nanstd(thickness_px[valid]),
            )

        # Step 2: Update Parquet with new columns
        logger.info("Step 2: Adding columns to Parquet ...")
        table = pq.read_table(cfg["parquet"])
        completeness = table.column("membrane_completeness").to_numpy()

        updated_table = update_parquet(
            cfg["parquet"], thickness_px, raw_dab, completeness,
        )
        logger.info(
            "  Parquet updated: %d columns, %d rows",
            updated_table.num_columns,
            updated_table.num_rows,
        )

        # Step 3: Compute summary
        summary = compute_summary(slide_id, updated_table)
        all_summaries.append(summary)

        # Step 4: Generate heatmaps
        logger.info("Step 4: Generating heatmaps for %s ...", slide_id)
        heatmap_paths = make_composite_heatmap(
            cfg["slide"], updated_table, slide_id,
        )
        all_heatmap_paths.extend(heatmap_paths)

    # Print results
    print_summary_table(all_summaries)

    # Save summary JSON
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EVAL_DIR / "membrane_thickness_results.json"
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "n_rays": N_RAYS,
            "ray_extent_px": RAY_EXTENT_PX,
            "min_dab_signal": MIN_DAB_SIGNAL,
            "pixel_size_um": PIXEL_SIZE_UM,
            "thickness_threshold_px": THICKNESS_THRESHOLD_PX,
            "completeness_threshold": COMPLETENESS_THRESHOLD,
            "dab_thresholds": {
                "negative": DAB_NEGATIVE_THRESHOLD,
                "weak": DAB_WEAK_THRESHOLD,
                "strong": DAB_STRONG_THRESHOLD,
            },
        },
        "slides": {s["slide_id"]: s for s in all_summaries},
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Summary JSON saved: %s", json_path)

    # Report heatmap files
    print("\nGenerated files:")
    for p in all_heatmap_paths:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<45} {size_kb:>8.0f} KB")

    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {elapsed_total:.0f}s ({elapsed_total / 60:.1f}min)")


if __name__ == "__main__":
    main()
