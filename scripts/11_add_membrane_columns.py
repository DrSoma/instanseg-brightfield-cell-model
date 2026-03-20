#!/home/fernandosoto/claudin18_venv/bin/python
"""Add membrane measurement columns to cohort parquet files.

For each slide in /tmp/cohort_v1/cell_data/, this script:
1. Opens the corresponding whole-slide image with openslide
2. For each cell, extracts the membrane ring (dilated - eroded polygon mask)
3. Measures DAB optical density at the membrane ring via stain deconvolution
4. Applies oriented bar-filter for directional membrane detection
5. Computes membrane completeness (fraction of boundary with DAB > threshold)
6. Estimates membrane thickness (FWHM of radial DAB profile)
7. Assigns a preliminary composite grade (UNCALIBRATED)

New columns written back to each parquet:
  - membrane_ring_dab (float64)
  - membrane_completeness (float64)
  - membrane_thickness_px (float64)
  - membrane_thickness_um (float64)
  - raw_membrane_dab (float64)
  - cldn18_composite_grade (int8): 0/1/2/3 -- UNCALIBRATED
  - thresholds_calibrated (bool): False for all rows

Usage:
    python scripts/11_add_membrane_columns.py
    python scripts/11_add_membrane_columns.py --cohort-dir /tmp/cohort_v1/cell_data
    python scripts/11_add_membrane_columns.py --workers 4
    python scripts/11_add_membrane_columns.py --slide-dir "/media/fernandosoto/DATA/CLDN18 slides"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Headless environment -- set before any Qt/matplotlib import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import numpy as np

logger = logging.getLogger("add_membrane_columns")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stain deconvolution vectors (calibrated for CLDN18 H-DAB slides)
STAIN_HEMATOXYLIN = [0.786, 0.593, 0.174]
STAIN_DAB = [0.215, 0.422, 0.881]
STAIN_RESIDUAL = [0.547, -0.799, 0.249]

# Bar-filter parameters (validated in comprehensive_membrane_validation.py)
BAR_N_ORIENTATIONS = 8
BAR_KSIZE = 25
BAR_SIGMA_LONG = 5.0
BAR_SIGMA_SHORT = 1.0

# Clinical thresholds for composite grade (PRELIMINARY / UNCALIBRATED)
# Grade 0: DAB < 0.10   (negative)
# Grade 1: 0.10 <= DAB < 0.20  (weak / 1+)
# Grade 2: 0.20 <= DAB < 0.35  (moderate / 2+)
# Grade 3: DAB >= 0.35  (strong / 3+)
GRADE_THRESHOLDS = (0.10, 0.20, 0.35)

# Slide directories to search (in priority order)
DEFAULT_SLIDE_DIRS = [
    Path("/media/fernandosoto/DATA/CLDN18 slides"),
    Path("/pathodata/Claudin18_project/slides"),
]

# Membrane ring morphology: erosion kernel and iterations for ~10px ring
RING_ERODE_KSIZE = 5
RING_ERODE_ITERATIONS = 2

# Edge detection for completeness: 1-pixel boundary
EDGE_ERODE_KSIZE = 3
EDGE_ERODE_ITERATIONS = 1

# Minimum pixels in ring/cell to consider valid
MIN_RING_PIXELS = 3
MIN_CELL_PIXELS = 10

# FWHM membrane thickness: radial profile parameters
FWHM_N_RADIAL_BINS = 20  # number of concentric shells for profile
FWHM_MAX_RADIUS_PX = 15  # max outward distance from boundary in pixels

# DAB positivity threshold for completeness
DAB_POSITIVE_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Stain deconvolution
# ---------------------------------------------------------------------------

def _build_stain_matrix() -> np.ndarray:
    """Build and row-normalize the H-DAB stain matrix."""
    matrix = np.array(
        [STAIN_HEMATOXYLIN, STAIN_DAB, STAIN_RESIDUAL], dtype=np.float64
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _extract_dab(tile_rgb: np.ndarray, inv_matrix: np.ndarray) -> np.ndarray:
    """Extract DAB optical density from an RGB tile.

    Args:
        tile_rgb: uint8 (H, W, 3) RGB image.
        inv_matrix: precomputed inverse stain matrix (3x3).

    Returns:
        (H, W) float32 DAB optical density, clipped >= 0.
    """
    od = -np.log(np.clip(tile_rgb.astype(np.float64) / 255.0, 1 / 255, 1))
    return np.clip((od @ inv_matrix)[:, :, 1], 0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# Bar-filter membrane detection
# ---------------------------------------------------------------------------

def _build_bar_kernels() -> np.ndarray:
    """Build oriented Gaussian bar-filter kernels (numpy, CPU).

    Returns:
        (N_orientations, ksize, ksize) float64 zero-mean kernels.
    """
    half = BAR_KSIZE // 2
    y, x = np.mgrid[-half: half + 1, -half: half + 1].astype(np.float64)
    kernels = []
    for i in range(BAR_N_ORIENTATIONS):
        theta = i * np.pi / BAR_N_ORIENTATIONS
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t
        g = np.exp(
            -0.5 * (x_rot ** 2 / BAR_SIGMA_LONG ** 2 + y_rot ** 2 / BAR_SIGMA_SHORT ** 2)
        )
        g -= g.mean()
        g /= np.abs(g).sum() + 1e-10
        kernels.append(g)
    return np.stack(kernels)


def _apply_bar_filters(dab: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    """Apply bar-filter bank to DAB image, returning max response.

    Args:
        dab: (H, W) float32 DAB concentration.
        kernels: (N, ksize, ksize) filter kernels.

    Returns:
        (H, W) float32 max bar-filter response across orientations.
    """
    max_resp = np.full_like(dab, -np.inf)
    for k in kernels:
        resp = cv2.filter2D(dab, cv2.CV_32F, k.astype(np.float32))
        np.maximum(max_resp, resp, out=max_resp)
    return max_resp


# ---------------------------------------------------------------------------
# Composite grade assignment (UNCALIBRATED)
# ---------------------------------------------------------------------------

def _assign_grade(membrane_dab: float) -> int:
    """Assign preliminary composite grade 0/1/2/3 from membrane DAB OD.

    Thresholds are UNCALIBRATED and for research use only.
    """
    t1, t2, t3 = GRADE_THRESHOLDS
    if np.isnan(membrane_dab):
        return 0
    if membrane_dab < t1:
        return 0
    elif membrane_dab < t2:
        return 1
    elif membrane_dab < t3:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Membrane thickness (FWHM approach)
# ---------------------------------------------------------------------------

def _estimate_membrane_thickness(
    dab_crop: np.ndarray,
    cell_mask_crop: np.ndarray,
) -> float:
    """Estimate membrane thickness using FWHM of radial DAB profile.

    Builds concentric rings outward from the cell interior and measures
    the DAB profile; the FWHM of the peak gives the thickness in pixels.

    Args:
        dab_crop: (h, w) float32 DAB OD in cell bounding-box region.
        cell_mask_crop: (h, w) uint8 binary mask of the cell.

    Returns:
        Membrane thickness in pixels (float). NaN if unmeasurable.
    """
    if cell_mask_crop.sum() < MIN_CELL_PIXELS:
        return np.nan

    # Distance transform from cell interior boundary
    # Positive inside, negative outside
    dist_inside = cv2.distanceTransform(cell_mask_crop, cv2.DIST_L2, 5)
    inverted = 1 - cell_mask_crop
    dist_outside = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)

    # Signed distance: positive = inside cell, negative = outside
    # We want the profile centered on the cell boundary (distance=0)
    signed_dist = dist_inside - dist_outside

    # Build radial profile: sample DAB at distance bins from boundary
    # Range: -FWHM_MAX_RADIUS_PX (outside) to +FWHM_MAX_RADIUS_PX (inside)
    bin_edges = np.linspace(
        -FWHM_MAX_RADIUS_PX, FWHM_MAX_RADIUS_PX, FWHM_N_RADIAL_BINS + 1
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    profile = np.full(FWHM_N_RADIAL_BINS, np.nan)

    for b in range(FWHM_N_RADIAL_BINS):
        mask = (signed_dist >= bin_edges[b]) & (signed_dist < bin_edges[b + 1])
        if mask.sum() >= 2:
            profile[b] = float(dab_crop[mask].mean())

    # Interpolate NaN gaps if possible
    valid = np.isfinite(profile)
    if valid.sum() < 3:
        return np.nan

    # Simple interpolation over NaN gaps
    profile_interp = np.interp(
        bin_centers, bin_centers[valid], profile[valid]
    )

    # Find FWHM
    peak_val = profile_interp.max()
    if peak_val < DAB_POSITIVE_THRESHOLD:
        return np.nan  # No significant DAB signal

    half_max = peak_val / 2.0
    above_half = profile_interp >= half_max

    # Find first and last crossing of half-max
    crossings = np.diff(above_half.astype(int))
    rise = np.where(crossings == 1)[0]
    fall = np.where(crossings == -1)[0]

    if len(rise) == 0 and len(fall) == 0:
        # Entirely above half-max
        return float(bin_centers[-1] - bin_centers[0])

    left = bin_centers[rise[0] + 1] if len(rise) > 0 else bin_centers[0]
    right = bin_centers[fall[-1]] if len(fall) > 0 else bin_centers[-1]

    fwhm = abs(right - left)
    return float(fwhm) if fwhm > 0 else np.nan


# ---------------------------------------------------------------------------
# Slide discovery
# ---------------------------------------------------------------------------

def _find_slide(slide_id: str, slide_dirs: list[Path]) -> Path | None:
    """Search for a slide file (.ndpi or .svs) in the given directories."""
    for sdir in slide_dirs:
        if not sdir.exists():
            continue
        for ext in (".ndpi", ".svs", ".tiff", ".tif"):
            candidate = sdir / f"{slide_id}{ext}"
            if candidate.exists():
                return candidate
    return None


# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------

def process_slide(
    parquet_path: Path,
    slide_dirs: list[Path],
    stain_inv: np.ndarray,
    bar_kernels: np.ndarray,
    slide_index: int,
    total_slides: int,
) -> dict[str, Any]:
    """Add membrane columns to a single slide's parquet file.

    Returns a summary dict with slide_id, n_cells, n_measured, elapsed.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import from_wkb

    slide_id = parquet_path.stem.replace("_cells", "")
    prefix = f"[{slide_index}/{total_slides}] {slide_id}"

    result: dict[str, Any] = {
        "slide_id": slide_id,
        "parquet": str(parquet_path),
        "status": "skipped",
        "n_cells": 0,
        "n_measured": 0,
        "elapsed_s": 0.0,
    }

    t0 = time.time()

    # ── Load parquet ──────────────────────────────────────────────────────
    try:
        table = pq.read_table(parquet_path)
    except Exception as exc:
        logger.error("%s: failed to read parquet: %s", prefix, exc)
        result["status"] = "error_read"
        return result

    n_cells = table.num_rows
    result["n_cells"] = n_cells

    # Idempotency: skip if already has membrane columns
    if "membrane_ring_dab" in table.column_names:
        logger.info("%s: already has membrane columns, skipping", prefix)
        result["status"] = "already_done"
        result["elapsed_s"] = time.time() - t0
        return result

    # ── Find the slide ────────────────────────────────────────────────────
    slide_path = _find_slide(slide_id, slide_dirs)
    if slide_path is None:
        logger.warning("%s: slide not found in any directory, skipping", prefix)
        result["status"] = "slide_not_found"
        result["elapsed_s"] = time.time() - t0
        return result

    logger.info("%s: processing %d cells (slide: %s)", prefix, n_cells, slide_path.name)

    # ── Read cell data ────────────────────────────────────────────────────
    polygon_wkb_col = table.column("polygon_wkb").to_pylist()
    cx_col = table.column("centroid_x").to_numpy()
    cy_col = table.column("centroid_y").to_numpy()

    # Initialize output arrays
    membrane_ring_dab = np.full(n_cells, np.nan, dtype=np.float64)
    membrane_completeness = np.full(n_cells, np.nan, dtype=np.float64)
    membrane_thickness_px = np.full(n_cells, np.nan, dtype=np.float64)
    raw_membrane_dab = np.full(n_cells, np.nan, dtype=np.float64)

    # ── Open slide ────────────────────────────────────────────────────────
    try:
        import openslide
    except ImportError:
        logger.error("%s: openslide not available", prefix)
        result["status"] = "error_openslide"
        result["elapsed_s"] = time.time() - t0
        return result

    try:
        slide = openslide.OpenSlide(str(slide_path))
    except Exception as exc:
        logger.error("%s: failed to open slide: %s", prefix, exc)
        result["status"] = "error_open"
        result["elapsed_s"] = time.time() - t0
        return result

    try:
        # Determine optimal read level (~0.5 um/px)
        base_mpp = float(slide.properties.get("openslide.mpp-x", 0.25))
        target_mpp = 0.5
        best_level = 0
        for lv in range(slide.level_count):
            lv_mpp = base_mpp * slide.level_downsamples[lv]
            if lv_mpp <= target_mpp * 1.2:
                best_level = lv
        level_ds = slide.level_downsamples[best_level]
        actual_mpp = base_mpp * level_ds

        # Tile-based grouping (tile step = (512 - 2*80) * level_downsample)
        tile_step_l0 = (512 - 2 * 80) * level_ds
        tile_size_l0 = 512 * level_ds
        padding_l0 = 80 * level_ds

        # Group cells into tile bins
        bins: dict[tuple[int, int], list[int]] = {}
        for i in range(n_cells):
            gx = int(cx_col[i] / tile_step_l0)
            gy = int(cy_col[i] / tile_step_l0)
            key = (gx, gy)
            if key not in bins:
                bins[key] = []
            bins[key].append(i)

        n_tiles = len(bins)
        processed_tiles = 0
        processed_cells = 0

        for (gx, gy), cell_indices in bins.items():
            processed_tiles += 1
            if processed_tiles % 500 == 0 or processed_tiles == n_tiles:
                logger.info(
                    "%s: tile %d/%d (%d cells measured so far)",
                    prefix, processed_tiles, n_tiles, processed_cells,
                )

            # Read tile region with padding
            x0_l0 = int(gx * tile_step_l0 - padding_l0)
            y0_l0 = int(gy * tile_step_l0 - padding_l0)
            read_w = int(tile_size_l0 + 2 * padding_l0)
            read_h = int(tile_size_l0 + 2 * padding_l0)

            # Clamp to slide bounds
            slide_w, slide_h = slide.dimensions
            x0_l0 = max(0, x0_l0)
            y0_l0 = max(0, y0_l0)
            read_w = min(read_w, slide_w - x0_l0)
            read_h = min(read_h, slide_h - y0_l0)

            if read_w <= 0 or read_h <= 0:
                continue

            # Read at best_level
            read_w_lv = max(1, int(read_w / level_ds))
            read_h_lv = max(1, int(read_h / level_ds))

            try:
                region_img = slide.read_region(
                    (x0_l0, y0_l0), best_level, (read_w_lv, read_h_lv)
                )
                tile_rgb = np.array(region_img.convert("RGB"))
            except Exception as exc:
                logger.debug(
                    "%s: failed to read tile (%d,%d): %s", prefix, gx, gy, exc
                )
                continue

            # DAB extraction
            dab = _extract_dab(tile_rgb, stain_inv)

            # Bar-filter response
            bar_resp = _apply_bar_filters(dab, bar_kernels)
            bar_positive = bar_resp > 0

            # Process each cell in this tile
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

                # Cell bounds in tile-local level coordinates
                bx0, by0, bx1, by1 = poly.bounds
                lx0 = int((bx0 - x0_l0) / level_ds)
                ly0 = int((by0 - y0_l0) / level_ds)
                lx1 = int((bx1 - x0_l0) / level_ds) + 1
                ly1 = int((by1 - y0_l0) / level_ds) + 1

                # Clamp to tile
                lx0 = max(0, lx0)
                ly0 = max(0, ly0)
                lx1 = min(read_w_lv, lx1)
                ly1 = min(read_h_lv, ly1)

                if lx1 <= lx0 or ly1 <= ly0:
                    continue

                # Rasterize cell polygon to mask in local coordinates
                try:
                    ext_coords = np.array(poly.exterior.coords)
                except Exception:
                    continue
                local_coords = (ext_coords - np.array([x0_l0, y0_l0])) / level_ds
                local_pts = local_coords.astype(np.int32)

                cell_mask = np.zeros((read_h_lv, read_w_lv), dtype=np.uint8)
                cv2.fillPoly(cell_mask, [local_pts], 1)

                # Crop to cell bounding box
                cell_crop = cell_mask[ly0:ly1, lx0:lx1]
                if cell_crop.sum() < MIN_CELL_PIXELS:
                    continue

                dab_crop = dab[ly0:ly1, lx0:lx1]
                bar_crop = bar_resp[ly0:ly1, lx0:lx1]
                bar_pos_crop = bar_positive[ly0:ly1, lx0:lx1]

                # --- Membrane ring: dilated - eroded cell mask ---
                eroded = cv2.erode(
                    cell_crop,
                    np.ones((RING_ERODE_KSIZE, RING_ERODE_KSIZE), np.uint8),
                    iterations=RING_ERODE_ITERATIONS,
                )
                ring_mask = (cell_crop > 0) & (eroded == 0)

                if ring_mask.sum() < MIN_RING_PIXELS:
                    continue

                # --- Membrane ring DAB (bar-filter weighted) ---
                ring_dab = dab_crop[ring_mask]
                ring_bar = np.clip(bar_crop[ring_mask], 0, None)

                if ring_bar.sum() > 0:
                    weighted_dab = float(np.average(ring_dab, weights=ring_bar))
                else:
                    weighted_dab = float(ring_dab.mean())

                membrane_ring_dab[cell_idx] = weighted_dab

                # Raw (unweighted) mean DAB at membrane ring
                raw_membrane_dab[cell_idx] = float(ring_dab.mean())

                # --- Membrane completeness ---
                edge_mask_src = cell_crop.copy()
                edge_eroded = cv2.erode(
                    edge_mask_src,
                    np.ones((EDGE_ERODE_KSIZE, EDGE_ERODE_KSIZE), np.uint8),
                    iterations=EDGE_ERODE_ITERATIONS,
                )
                edge = (edge_mask_src > 0) & (edge_eroded == 0)
                if edge.sum() > 0:
                    # Completeness = fraction of boundary pixels with
                    # positive bar-filter AND DAB above threshold
                    edge_bar_pos = bar_pos_crop[edge]
                    edge_dab = dab_crop[edge]
                    membrane_completeness[cell_idx] = float(
                        ((edge_bar_pos) & (edge_dab > DAB_POSITIVE_THRESHOLD)).sum()
                        / edge.sum()
                    )

                # --- Membrane thickness (FWHM) ---
                # Use the full cell region for distance transform
                thickness_px = _estimate_membrane_thickness(dab_crop, cell_crop)
                membrane_thickness_px[cell_idx] = thickness_px

                processed_cells += 1

    finally:
        slide.close()

    # ── Derived columns ───────────────────────────────────────────────────
    # Membrane thickness in microns
    membrane_thickness_um = membrane_thickness_px * actual_mpp

    # Composite grade (UNCALIBRATED)
    grades = np.array(
        [_assign_grade(v) for v in membrane_ring_dab], dtype=np.int8
    )

    # All thresholds flagged as uncalibrated
    calibrated = np.zeros(n_cells, dtype=bool)

    # ── Write back to parquet ─────────────────────────────────────────────
    new_table = table
    new_table = new_table.append_column(
        "membrane_ring_dab",
        pa.array(membrane_ring_dab, type=pa.float64()),
    )
    new_table = new_table.append_column(
        "membrane_completeness",
        pa.array(membrane_completeness, type=pa.float64()),
    )
    new_table = new_table.append_column(
        "membrane_thickness_px",
        pa.array(membrane_thickness_px, type=pa.float64()),
    )
    new_table = new_table.append_column(
        "membrane_thickness_um",
        pa.array(membrane_thickness_um, type=pa.float64()),
    )
    new_table = new_table.append_column(
        "raw_membrane_dab",
        pa.array(raw_membrane_dab, type=pa.float64()),
    )
    new_table = new_table.append_column(
        "cldn18_composite_grade",
        pa.array(grades, type=pa.int8()),
    )
    new_table = new_table.append_column(
        "thresholds_calibrated",
        pa.array(calibrated, type=pa.bool_()),
    )

    pq.write_table(new_table, parquet_path, compression="zstd")

    elapsed = time.time() - t0
    n_measured = int(np.isfinite(membrane_ring_dab).sum())

    logger.info(
        "%s: done -- %d/%d cells measured (%.1f%%), %.1fs",
        prefix,
        n_measured,
        n_cells,
        100 * n_measured / max(n_cells, 1),
        elapsed,
    )

    result.update({
        "status": "ok",
        "n_measured": n_measured,
        "elapsed_s": elapsed,
        "actual_mpp": actual_mpp,
        "n_tiles": n_tiles,
    })
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add membrane measurement columns to cohort parquet files. "
                    "All grades are UNCALIBRATED -- research use only.",
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=Path("/tmp/cohort_v1/cell_data"),
        help="Directory containing *_cells.parquet files (default: /tmp/cohort_v1/cell_data)",
    )
    parser.add_argument(
        "--slide-dir",
        type=str,
        nargs="*",
        default=None,
        help="Directories to search for slide files. "
             "Default: /media/fernandosoto/DATA/CLDN18 slides, /pathodata/Claudin18_project/slides",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker threads (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be processed without actually running",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve slide directories
    if args.slide_dir:
        slide_dirs = [Path(d) for d in args.slide_dir]
    else:
        slide_dirs = DEFAULT_SLIDE_DIRS

    existing_dirs = [d for d in slide_dirs if d.exists()]
    logger.info("Slide directories: %s", [str(d) for d in existing_dirs])
    if not existing_dirs:
        logger.warning(
            "No slide directories found. Slides at: %s",
            [str(d) for d in slide_dirs],
        )

    # Discover parquet files
    cohort_dir = args.cohort_dir
    if not cohort_dir.exists():
        logger.error("Cohort directory does not exist: %s", cohort_dir)
        sys.exit(1)

    parquet_files = sorted(cohort_dir.glob("*_cells.parquet"))
    if not parquet_files:
        logger.error("No *_cells.parquet files found in %s", cohort_dir)
        sys.exit(1)

    logger.info(
        "Found %d parquet files in %s", len(parquet_files), cohort_dir
    )

    if args.dry_run:
        for pf in parquet_files:
            sid = pf.stem.replace("_cells", "")
            sp = _find_slide(sid, slide_dirs)
            status = f"slide found: {sp}" if sp else "SLIDE NOT FOUND"
            print(f"  {pf.name} -> {status}")
        return

    # Precompute stain deconvolution and bar-filter
    logger.info("Building stain deconvolution matrix and bar-filter kernels...")
    stain_matrix = _build_stain_matrix()
    stain_inv = np.linalg.inv(stain_matrix)
    bar_kernels = _build_bar_kernels()
    logger.info(
        "Bar-filter: %d orientations, ksize=%d, sigma_long=%.1f, sigma_short=%.1f",
        BAR_N_ORIENTATIONS, BAR_KSIZE, BAR_SIGMA_LONG, BAR_SIGMA_SHORT,
    )

    total = len(parquet_files)
    logger.info("=" * 70)
    logger.info(
        "Starting membrane column addition for %d slides (%d workers)",
        total,
        args.workers,
    )
    logger.info("Thresholds (UNCALIBRATED): %s", GRADE_THRESHOLDS)
    logger.info("=" * 70)

    # Process slides
    t_start = time.time()
    results: list[dict[str, Any]] = []

    if args.workers <= 1:
        # Sequential
        for idx, pf in enumerate(parquet_files, 1):
            res = process_slide(
                pf, slide_dirs, stain_inv, bar_kernels, idx, total
            )
            results.append(res)
    else:
        # Parallel with ThreadPoolExecutor
        # Note: openslide read_region releases the GIL, so threads help
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for idx, pf in enumerate(parquet_files, 1):
                fut = executor.submit(
                    process_slide,
                    pf, slide_dirs, stain_inv, bar_kernels, idx, total,
                )
                futures[fut] = pf

            for fut in as_completed(futures):
                pf = futures[fut]
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as exc:
                    sid = pf.stem.replace("_cells", "")
                    logger.error("Unhandled error for %s: %s", sid, exc)
                    results.append({
                        "slide_id": sid,
                        "status": "error_unhandled",
                        "error": str(exc),
                    })

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_skip = sum(1 for r in results if r.get("status") == "already_done")
    n_notfound = sum(1 for r in results if r.get("status") == "slide_not_found")
    n_error = sum(
        1 for r in results
        if r.get("status", "").startswith("error")
    )
    total_cells = sum(r.get("n_cells", 0) for r in results)
    total_measured = sum(r.get("n_measured", 0) for r in results)

    logger.info("=" * 70)
    logger.info("MEMBRANE COLUMN ADDITION COMPLETE")
    logger.info("  Total slides:      %d", total)
    logger.info("  Processed (ok):    %d", n_ok)
    logger.info("  Already done:      %d", n_skip)
    logger.info("  Slide not found:   %d", n_notfound)
    logger.info("  Errors:            %d", n_error)
    logger.info("  Total cells:       %d", total_cells)
    logger.info("  Total measured:    %d", total_measured)
    logger.info("  Measurement rate:  %.1f%%",
                100 * total_measured / max(total_cells, 1))
    logger.info("  Total time:        %.1fs", elapsed_total)
    logger.info("=" * 70)
    logger.info(
        "WARNING: All composite grades are UNCALIBRATED. "
        "Do NOT use for clinical decisions."
    )

    # Print problem slides
    problem_slides = [r for r in results if r.get("status") not in ("ok", "already_done")]
    if problem_slides:
        logger.warning("Problem slides:")
        for r in problem_slides:
            logger.warning(
                "  %s: %s", r.get("slide_id", "?"), r.get("status", "?")
            )


if __name__ == "__main__":
    main()
