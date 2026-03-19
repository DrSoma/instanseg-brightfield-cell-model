#!/usr/bin/env python3
"""Run pipeline comparison: baseline vs our model on BC control slides.

Executes the existing instanseg_batched.py pipeline on BC_ClassII and
BC_ClassIII slides using both the baseline (brightfield_nuclei) and our
trained model (brightfield_cells_nuclei), then compares cell counts,
H-scores, and DAB intensity.  Adds membrane-ring DAB measurement columns
to the v2 (our model) output using the validated bar-filter approach.

Outputs to separate directories to avoid overwriting baseline results.

Environment variables (set automatically by this script):
    CUDA_VISIBLE_DEVICES=1  (GPU 0 has SegFormer)
    QT_QPA_PLATFORM=offscreen
    MPLBACKEND=Agg
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Usage:
    python scripts/run_pipeline_comparison.py [--skip-baseline] [--skip-v2] [--skip-membrane]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Pipeline paths
PIPELINE_DIR = Path(
    "/home/fernandosoto/Documents/cldn18-pathology-pipeline-main (3)/"
    "cldn18-pathology-pipeline-main"
)
PIPELINE_SCRIPT = PIPELINE_DIR / "scripts" / "instanseg_batched.py"

# Our exported model
OUR_MODEL_PT = PROJECT_ROOT / "models" / "exported" / "brightfield_cells_nuclei.pt"

# Slide paths
SLIDES_DIR = Path("/tmp/bc_slides")
SLIDES = ["BC_ClassII", "BC_ClassIII"]
SLIDE_PATHS = {s: SLIDES_DIR / f"{s}.ndpi" for s in SLIDES}

# Region annotation source
REGION_ANNOTATION_DIR = Path(
    "/pathodata/Claudin18_project/preprocessing/region_annotations"
)

# Existing tissue polygon / cell data (for bootstrapping project dirs)
EXISTING_PROJECT_DIR = Path("/pathodata/Claudin18_project")

# Baseline results for comparison
BASELINE_RESULTS_CSV = Path(
    "/pathodata/Claudin18_project/exports/cldn18_region_summary.csv"
)

# Output
COMPARISON_ROOT = Path("/tmp/pipeline_comparison")
V1_PROJECT = COMPARISON_ROOT / "v1_baseline"
V2_PROJECT = COMPARISON_ROOT / "v2_cells_nuclei"
EVAL_OUTPUT = PROJECT_ROOT / "evaluation" / "pipeline_comparison"

# InstanSeg bioimageio model cache (inside the venv)
INSTANSEG_BIOIMAGEIO_DIR = Path(
    "/home/fernandosoto/claudin18_venv/lib/python3.12/"
    "site-packages/instanseg/bioimageio_models"
)

# Python interpreter — must be the claudin18 venv (has instanseg, pyarrow, etc.)
VENV_PYTHON = "/home/fernandosoto/claudin18_venv/bin/python"

# Stain deconvolution vectors (calibrated for CLDN18 slides)
STAIN_HEMATOXYLIN = [0.786, 0.593, 0.174]
STAIN_DAB = [0.215, 0.422, 0.881]
STAIN_RESIDUAL = [0.547, -0.799, 0.249]

# Clinical thresholds (whole-cell DAB OD)
CLINICAL_THRESHOLDS = (0.10, 0.20, 0.35)

# Bar-filter parameters (validated in comprehensive_membrane_validation.py)
BAR_N_ORIENTATIONS = 8
BAR_KSIZE = 25
BAR_SIGMA_LONG = 5.0
BAR_SIGMA_SHORT = 1.0

# Environment for subprocess runs
PIPELINE_ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "1",
    "QT_QPA_PLATFORM": "offscreen",
    "MPLBACKEND": "Agg",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


# ── Stain deconvolution (standalone, no project imports needed) ──────────────


def _build_stain_matrix() -> np.ndarray:
    """Build and row-normalize the H-DAB stain matrix."""
    matrix = np.array(
        [STAIN_HEMATOXYLIN, STAIN_DAB, STAIN_RESIDUAL], dtype=np.float64
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _extract_dab_cpu(tile_rgb: np.ndarray, inv_matrix: np.ndarray) -> np.ndarray:
    """Extract DAB concentration from an RGB tile (CPU path).

    Args:
        tile_rgb: uint8 (H, W, 3) RGB image.
        inv_matrix: precomputed inverse stain matrix (3x3).

    Returns:
        (H, W) float32 DAB optical density, values >= 0.
    """
    od = -np.log(np.clip(tile_rgb.astype(np.float64) / 255.0, 1 / 255, 1))
    return np.clip((od @ inv_matrix)[:, :, 1], 0, None).astype(np.float32)


# ── Bar-filter membrane detection (standalone, torch-based) ──────────────────


def _build_bar_kernels() -> np.ndarray:
    """Build oriented Gaussian bar-filter kernels (numpy, CPU).

    Returns:
        (N_orientations, ksize, ksize) float64 zero-mean kernels.
    """
    half = BAR_KSIZE // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
    kernels = []
    for i in range(BAR_N_ORIENTATIONS):
        theta = i * np.pi / BAR_N_ORIENTATIONS
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t
        g = np.exp(
            -0.5 * (x_rot**2 / BAR_SIGMA_LONG**2 + y_rot**2 / BAR_SIGMA_SHORT**2)
        )
        g -= g.mean()
        g /= np.abs(g).sum() + 1e-10
        kernels.append(g)
    return np.stack(kernels)


def _apply_bar_filters_cpu(
    dab: np.ndarray, kernels: np.ndarray
) -> np.ndarray:
    """Apply bar-filter orientations to a DAB image on CPU.

    Uses cv2.filter2D for each orientation, takes max across all.

    Args:
        dab: (H, W) float32 DAB concentration.
        kernels: (N, ksize, ksize) filter kernels.

    Returns:
        (H, W) float32 max bar-filter response.
    """
    max_resp = np.full_like(dab, -np.inf)
    for k in kernels:
        resp = cv2.filter2D(dab, cv2.CV_32F, k.astype(np.float32))
        np.maximum(max_resp, resp, out=max_resp)
    return max_resp


# ── H-score computation ─────────────────────────────────────────────────────


def classify_dab(dab_value: float, thresholds: tuple[float, ...]) -> int:
    """Classify DAB OD into 0/1+/2+/3+."""
    t1, t2, t3 = thresholds
    if dab_value < t1:
        return 0
    elif dab_value < t2:
        return 1
    elif dab_value < t3:
        return 2
    return 3


def compute_hscore(classes: list[int] | np.ndarray) -> float:
    """H-score: 1*%1+ + 2*%2+ + 3*%3+.  Range 0-300."""
    c = np.asarray(classes)
    if len(c) == 0:
        return 0.0
    n = len(c)
    return float(
        1 * (c == 1).sum() / n * 100
        + 2 * (c == 2).sum() / n * 100
        + 3 * (c == 3).sum() / n * 100
    )


def compute_pct_positive(classes: list[int] | np.ndarray) -> float:
    """CLDN18.2 positivity: % cells with 2+ or 3+."""
    c = np.asarray(classes)
    if len(c) == 0:
        return 0.0
    return float((c >= 2).sum() / len(c) * 100)


# ── Project directory setup ──────────────────────────────────────────────────


def setup_project_dir(project_dir: Path) -> None:
    """Create a project directory with the structure instanseg_batched expects.

    Copies tissue polygons and region annotations from the existing project
    so the pipeline can find them.
    """
    project_dir.mkdir(parents=True, exist_ok=True)
    cell_data_dir = project_dir / "cell_data"
    cell_data_dir.mkdir(exist_ok=True)
    region_ann_dir = project_dir / "preprocessing" / "region_annotations"
    region_ann_dir.mkdir(parents=True, exist_ok=True)

    for slide_id in SLIDES:
        # Copy tissue polygons (required by pipeline to know where tissue is)
        src_tissue = EXISTING_PROJECT_DIR / "cell_data" / f"{slide_id}_tissue.geojson"
        dst_tissue = cell_data_dir / f"{slide_id}_tissue.geojson"
        if src_tissue.exists() and not dst_tissue.exists():
            shutil.copy2(src_tissue, dst_tissue)
            logger.info("  Copied tissue polygons: %s -> %s", src_tissue.name, dst_tissue)

        # Copy region annotations
        src_region = REGION_ANNOTATION_DIR / f"{slide_id}_regions.geojson"
        dst_region = region_ann_dir / f"{slide_id}_regions.geojson"
        if src_region.exists() and not dst_region.exists():
            shutil.copy2(src_region, dst_region)
            logger.info("  Copied region annotations: %s -> %s", src_region.name, dst_region)


def install_our_model() -> Path:
    """Install our model into InstanSeg's bioimageio_models directory.

    Creates the directory structure that InstanSeg's download_model() expects.
    For models NOT in model-index.json, the path is:
        {bioimageio_path}/brightfield_cells_nuclei/instanseg.pt
    (no version subfolder — download_model uses version=None for unlisted models)

    Returns:
        Path to the installed instanseg.pt file.
    """
    model_dir = INSTANSEG_BIOIMAGEIO_DIR / "brightfield_cells_nuclei"
    model_dir.mkdir(parents=True, exist_ok=True)
    installed_pt = model_dir / "instanseg.pt"

    if installed_pt.exists():
        # Check if it is up to date (same size)
        if installed_pt.stat().st_size == OUR_MODEL_PT.stat().st_size:
            logger.info("  Model already installed at %s (up to date)", installed_pt)
            return installed_pt

    shutil.copy2(OUR_MODEL_PT, installed_pt)
    logger.info("  Installed model: %s -> %s", OUR_MODEL_PT, installed_pt)
    return installed_pt


# ── Pipeline execution ───────────────────────────────────────────────────────


def run_pipeline(
    model_name: str,
    project_dir: Path,
    label: str,
    extra_args: list[str] | None = None,
) -> bool:
    """Run instanseg_batched.py as a subprocess.

    Args:
        model_name: InstanSeg model identifier.
        project_dir: Output project directory.
        label: Human-readable label (for logging).
        extra_args: Additional CLI arguments.

    Returns:
        True if the pipeline completed successfully.
    """
    cmd = [
        VENV_PYTHON,
        str(PIPELINE_SCRIPT),
        "--slides-dir", str(SLIDES_DIR),
        "--project-dir", str(project_dir),
        "--output-format", "parquet",
        "--device", "cuda:0",  # remapped by CUDA_VISIBLE_DEVICES=1
        "--model", model_name,
        "--batch-size", "128",
        "--tile-size", "512",
        "--padding", "80",
        "--dedup-mode", "hybrid",
        "--dedup-radius", "5",
        "--optimize", "off",
        "--no-skip-measurements",
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("=" * 60)
    logger.info("  Running pipeline: %s", label)
    logger.info("  Model: %s", model_name)
    logger.info("  Project dir: %s", project_dir)
    logger.info("  Command: %s", " ".join(cmd))
    logger.info("=" * 60)

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=PIPELINE_ENV,
        cwd=str(PIPELINE_DIR / "scripts"),
    )

    # Stream output in real time
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.rstrip("\n")
        print(f"  [{label}] {text}")

    return_code = proc.wait()
    elapsed = time.time() - t0

    if return_code == 0:
        logger.info("  %s completed in %.1fs", label, elapsed)
        return True
    else:
        logger.error(
            "  %s FAILED (exit code %d) after %.1fs",
            label, return_code, elapsed,
        )
        return False


# ── Read Parquet results ─────────────────────────────────────────────────────


def read_parquet_results(project_dir: Path) -> dict[str, Any]:
    """Read cell data Parquet files from a project directory.

    Returns:
        Dict mapping slide_id -> pyarrow.Table.
    """
    import pyarrow.parquet as pq

    cell_data_dir = project_dir / "cell_data"
    results = {}
    for slide_id in SLIDES:
        parquet_path = cell_data_dir / f"{slide_id}_cells.parquet"
        if parquet_path.exists():
            table = pq.read_table(parquet_path)
            results[slide_id] = table
            logger.info(
                "  Read %s: %d cells, %d columns",
                parquet_path.name, table.num_rows, table.num_columns,
            )
        else:
            logger.warning("  Missing: %s", parquet_path)
    return results


def read_baseline_csv() -> dict[str, dict[str, Any]]:
    """Read the existing pipeline's region summary CSV.

    Returns:
        Nested dict: {slide_name: {region: {metric: value}}}.
    """
    import csv

    results: dict[str, dict[str, Any]] = {}
    if not BASELINE_RESULTS_CSV.exists():
        logger.warning("Baseline CSV not found: %s", BASELINE_RESULTS_CSV)
        return results

    with open(BASELINE_RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide = row.get("slide_name", "").strip()
            region = row.get("region", "").strip()
            if not slide or not region:
                continue
            # Skip duplicate header rows (CSV has duplicated header line)
            if slide == "slide_name":
                continue
            try:
                total_cells = int(row.get("total_cells", 0))
                pct_positive = float(row.get("pct_positive", 0))
                h_score = float(row.get("h_score", 0))
                mean_dab = float(row.get("mean_dab_od", 0))
            except (ValueError, TypeError):
                continue
            if slide not in results:
                results[slide] = {}
            results[slide][region] = {
                "total_cells": total_cells,
                "pct_positive": pct_positive,
                "h_score": h_score,
                "mean_dab_od": mean_dab,
            }
    return results


# ── Membrane-ring DAB measurement (Step 6) ───────────────────────────────────


def add_membrane_ring_columns(
    project_dir: Path,
    progress_label: str = "v2",
) -> None:
    """Add membrane-ring DAB columns to Parquet files.

    For each cell in the Parquet output:
    1. Read the polygon_wkb to get the cell boundary.
    2. Read the corresponding region from the slide at ~0.5 um/px.
    3. Compute DAB via stain deconvolution.
    4. Apply bar-filter (8 oriented Gaussian kernels).
    5. Compute membrane_ring_dab = mean DAB weighted by bar-filter
       response within 10px of the cell boundary.
    6. Compute membrane_completeness = fraction of boundary pixels
       with positive bar-filter response.

    Writes membrane_ring_dab and membrane_completeness as new columns.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import from_wkb

    try:
        import openslide
    except ImportError:
        logger.error("openslide-python not installed; cannot add membrane columns")
        return

    stain_inv = np.linalg.inv(_build_stain_matrix())
    bar_kernels = _build_bar_kernels()

    for slide_id in SLIDES:
        slide_path = SLIDE_PATHS[slide_id]
        parquet_path = project_dir / "cell_data" / f"{slide_id}_cells.parquet"

        if not parquet_path.exists():
            logger.warning("  Skipping %s: no parquet file", slide_id)
            continue

        if not slide_path.exists():
            logger.warning("  Skipping %s: slide not found at %s", slide_id, slide_path)
            continue

        logger.info("  Adding membrane columns to %s ...", parquet_path.name)
        t0 = time.time()

        table = pq.read_table(parquet_path)
        n_cells = table.num_rows

        # Pre-check: skip if already done
        if "membrane_ring_dab" in table.column_names:
            logger.info("    Already has membrane_ring_dab, skipping")
            continue

        # Read polygon WKB and centroids
        polygon_wkb_col = table.column("polygon_wkb").to_pylist()
        cx_col = table.column("centroid_x").to_numpy()
        cy_col = table.column("centroid_y").to_numpy()

        membrane_ring_dab = np.full(n_cells, np.nan, dtype=np.float64)
        membrane_completeness = np.full(n_cells, np.nan, dtype=np.float64)

        slide = openslide.OpenSlide(str(slide_path))
        try:
            # Determine the optimal read level (~0.5 um/px)
            base_mpp = float(slide.properties.get("openslide.mpp-x", 0.25))
            target_mpp = 0.5
            best_level = 0
            for lv in range(slide.level_count):
                lv_mpp = base_mpp * slide.level_downsamples[lv]
                if lv_mpp <= target_mpp * 1.2:
                    best_level = lv
            level_ds = slide.level_downsamples[best_level]

            # Process cells in spatial batches (group by tile region)
            # For efficiency, group cells that are near each other and read
            # one tile region covering them.
            tile_size_l0 = 512 * level_ds  # ~256um at 0.5 um/px
            padding_l0 = 80 * level_ds

            # Group cells into tile bins
            bins: dict[tuple[int, int], list[int]] = {}
            for i in range(n_cells):
                gx = int(cx_col[i] / tile_size_l0)
                gy = int(cy_col[i] / tile_size_l0)
                key = (gx, gy)
                if key not in bins:
                    bins[key] = []
                bins[key].append(i)

            processed = 0
            from tqdm import tqdm as _tqdm

            for (gx, gy), cell_indices in _tqdm(
                bins.items(),
                desc=f"  [{progress_label}] {slide_id} membrane",
                unit="tile",
            ):
                # Read tile region with padding
                x0_l0 = int(gx * tile_size_l0 - padding_l0)
                y0_l0 = int(gy * tile_size_l0 - padding_l0)
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
                region_img = slide.read_region(
                    (x0_l0, y0_l0), best_level, (read_w_lv, read_h_lv)
                )
                tile_rgb = np.array(region_img.convert("RGB"))

                # DAB extraction
                dab = _extract_dab_cpu(tile_rgb, stain_inv)

                # Bar-filter
                bar_resp = _apply_bar_filters_cpu(dab, bar_kernels)
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

                    # Get cell bounds in tile-local coordinates
                    bx0, by0, bx1, by1 = poly.bounds
                    # Convert from level-0 to tile-local level coordinates
                    lx0 = int((bx0 - x0_l0) / level_ds)
                    ly0 = int((by0 - y0_l0) / level_ds)
                    lx1 = int((bx1 - x0_l0) / level_ds) + 1
                    ly1 = int((by1 - y0_l0) / level_ds) + 1

                    # Clamp to tile bounds
                    lx0 = max(0, lx0)
                    ly0 = max(0, ly0)
                    lx1 = min(read_w_lv, lx1)
                    ly1 = min(read_h_lv, ly1)

                    if lx1 <= lx0 or ly1 <= ly0:
                        continue

                    # Rasterize cell polygon to a mask in the local region
                    # Convert polygon coords to local pixel coords
                    ext_coords = np.array(poly.exterior.coords)
                    local_coords = (ext_coords - np.array([x0_l0, y0_l0])) / level_ds
                    local_pts = local_coords.astype(np.int32)

                    cell_mask = np.zeros((read_h_lv, read_w_lv), dtype=np.uint8)
                    cv2.fillPoly(cell_mask, [local_pts], 1)

                    # Crop to cell bounding box for efficiency
                    cell_crop = cell_mask[ly0:ly1, lx0:lx1]
                    if cell_crop.sum() < 10:
                        continue

                    dab_crop = dab[ly0:ly1, lx0:lx1]
                    bar_crop = bar_resp[ly0:ly1, lx0:lx1]
                    bar_pos_crop = bar_positive[ly0:ly1, lx0:lx1]

                    # Membrane zone: cell boundary (10px erosion ring)
                    eroded = cv2.erode(
                        cell_crop, np.ones((5, 5), np.uint8), iterations=2
                    )
                    ring_mask = (cell_crop > 0) & (eroded == 0)

                    if ring_mask.sum() < 3:
                        continue

                    # Weighted membrane DAB (bar response as weights)
                    ring_dab = dab_crop[ring_mask]
                    ring_bar = np.clip(bar_crop[ring_mask], 0, None)

                    if ring_bar.sum() > 0:
                        weighted_dab = float(np.average(ring_dab, weights=ring_bar))
                    else:
                        weighted_dab = float(ring_dab.mean())

                    membrane_ring_dab[cell_idx] = weighted_dab

                    # Completeness: fraction of boundary with positive bar response
                    edge_mask = cell_crop.copy()
                    edge_eroded = cv2.erode(
                        edge_mask, np.ones((3, 3), np.uint8), iterations=1
                    )
                    edge = (edge_mask > 0) & (edge_eroded == 0)
                    if edge.sum() > 0:
                        membrane_completeness[cell_idx] = float(
                            bar_pos_crop[edge].mean()
                        )

                    processed += 1

        finally:
            slide.close()

        # Write new columns back to Parquet
        # Read existing columns (except polygon_wkb to save memory),
        # then append new columns
        existing_cols = [c for c in table.column_names]
        new_table = table
        new_table = new_table.append_column(
            "membrane_ring_dab",
            pa.array(membrane_ring_dab, type=pa.float64()),
        )
        new_table = new_table.append_column(
            "membrane_completeness",
            pa.array(membrane_completeness, type=pa.float64()),
        )
        pq.write_table(new_table, parquet_path, compression="zstd")

        valid = np.isfinite(membrane_ring_dab)
        elapsed = time.time() - t0
        logger.info(
            "    %s: %d/%d cells measured (%.1f%%), %.1fs",
            slide_id,
            valid.sum(),
            n_cells,
            100 * valid.sum() / max(n_cells, 1),
            elapsed,
        )


# ── Comparison and reporting ─────────────────────────────────────────────────


def compare_results(
    v1_data: dict[str, Any],
    v2_data: dict[str, Any],
    baseline_csv: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare v1 (baseline) and v2 (our model) results.

    Returns a structured comparison dict suitable for JSON serialization.
    """
    import pyarrow.parquet as pq

    comparison: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slides": {},
    }

    for slide_id in SLIDES:
        slide_comp: dict[str, Any] = {"slide_id": slide_id, "regions": {}}

        # --- V1 (baseline) ---
        v1_table = v1_data.get(slide_id)
        v1_total = v1_table.num_rows if v1_table is not None else 0
        v1_regions: dict[str, int] = {}
        if v1_table is not None and "region" in v1_table.column_names:
            regions = v1_table.column("region").to_pylist()
            for r in regions:
                v1_regions[r] = v1_regions.get(r, 0) + 1

        # --- V2 (our model) ---
        v2_table = v2_data.get(slide_id)
        v2_total = v2_table.num_rows if v2_table is not None else 0
        v2_regions: dict[str, int] = {}
        if v2_table is not None and "region" in v2_table.column_names:
            regions = v2_table.column("region").to_pylist()
            for r in regions:
                v2_regions[r] = v2_regions.get(r, 0) + 1

        slide_comp["v1_total_cells"] = v1_total
        slide_comp["v2_total_cells"] = v2_total
        slide_comp["cell_count_ratio"] = (
            v2_total / v1_total if v1_total > 0 else float("nan")
        )

        # --- Existing pipeline baseline ---
        csv_data = baseline_csv.get(slide_id, {})
        slide_comp["csv_baseline"] = csv_data

        # --- Per-region comparison ---
        all_regions = sorted(set(list(v1_regions.keys()) + list(v2_regions.keys())))
        for region in all_regions:
            region_comp: dict[str, Any] = {
                "v1_cells": v1_regions.get(region, 0),
                "v2_cells": v2_regions.get(region, 0),
            }

            # V2 membrane-ring H-score (if membrane columns present)
            if v2_table is not None and "membrane_ring_dab" in v2_table.column_names:
                # Filter to this region
                region_col = v2_table.column("region").to_pylist()
                membrane_col = v2_table.column("membrane_ring_dab").to_numpy()

                region_mask = np.array([r == region for r in region_col])
                region_dab = membrane_col[region_mask]
                valid = np.isfinite(region_dab)
                region_dab_valid = region_dab[valid]

                if len(region_dab_valid) > 0:
                    classes = [
                        classify_dab(float(d), CLINICAL_THRESHOLDS)
                        for d in region_dab_valid
                    ]
                    region_comp["v2_membrane_ring"] = {
                        "n_measured": int(valid.sum()),
                        "mean_dab": float(region_dab_valid.mean()),
                        "std_dab": float(region_dab_valid.std()),
                        "h_score": compute_hscore(classes),
                        "pct_positive": compute_pct_positive(classes),
                        "pct_negative": float((np.array(classes) == 0).mean() * 100),
                        "pct_1plus": float((np.array(classes) == 1).mean() * 100),
                        "pct_2plus": float((np.array(classes) == 2).mean() * 100),
                        "pct_3plus": float((np.array(classes) == 3).mean() * 100),
                    }

            # V2 whole-cell DAB (if dab_mean column present)
            if v2_table is not None and "dab_mean" in v2_table.column_names:
                region_col = v2_table.column("region").to_pylist()
                dab_col = v2_table.column("dab_mean").to_numpy()
                region_mask = np.array([r == region for r in region_col])
                region_dab = dab_col[region_mask]
                valid = np.isfinite(region_dab)
                if valid.sum() > 0:
                    region_comp["v2_whole_cell_dab_mean"] = float(
                        region_dab[valid].mean()
                    )

            # CSV baseline for this region
            if region in csv_data:
                region_comp["csv_baseline"] = csv_data[region]

            slide_comp["regions"][region] = region_comp

        comparison["slides"][slide_id] = slide_comp

    return comparison


def print_comparison_table(comparison: dict[str, Any]) -> None:
    """Print a human-readable comparison table."""
    print()
    print("=" * 90)
    print("PIPELINE COMPARISON: BASELINE vs OUR MODEL")
    print("=" * 90)

    for slide_id, slide_data in comparison.get("slides", {}).items():
        print(f"\n{'─' * 90}")
        print(f"  Slide: {slide_id}")
        print(f"  V1 (baseline) cells: {slide_data.get('v1_total_cells', '?'):>10,}")
        print(f"  V2 (our model) cells: {slide_data.get('v2_total_cells', '?'):>9,}")
        ratio = slide_data.get("cell_count_ratio", float("nan"))
        if not (ratio != ratio):  # not NaN
            print(f"  Cell count ratio (v2/v1): {ratio:.3f}")
        print()

        # Per-region table
        regions = slide_data.get("regions", {})
        if regions:
            header = (
                f"  {'Region':<20} {'V1 cells':>10} {'V2 cells':>10} "
                f"{'CSV H-score':>12} {'V2 Ring H':>10} {'V2 Ring Pos%':>12}"
            )
            print(header)
            print(f"  {'─' * 78}")

            for region, rdata in regions.items():
                v1_cells = rdata.get("v1_cells", 0)
                v2_cells = rdata.get("v2_cells", 0)

                csv_h = "N/A"
                csv_base = rdata.get("csv_baseline", {})
                if csv_base:
                    csv_h = f"{csv_base.get('h_score', 0):.1f}"

                ring_h = "N/A"
                ring_pos = "N/A"
                ring_data = rdata.get("v2_membrane_ring", {})
                if ring_data:
                    ring_h = f"{ring_data.get('h_score', 0):.1f}"
                    ring_pos = f"{ring_data.get('pct_positive', 0):.1f}%"

                print(
                    f"  {region:<20} {v1_cells:>10,} {v2_cells:>10,} "
                    f"{csv_h:>12} {ring_h:>10} {ring_pos:>12}"
                )

    print(f"\n{'=' * 90}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline comparison: baseline vs our model on BC slides",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip running the baseline pipeline (use existing v1 results)",
    )
    parser.add_argument(
        "--skip-v2",
        action="store_true",
        help="Skip running the v2 pipeline (use existing v2 results)",
    )
    parser.add_argument(
        "--skip-membrane",
        action="store_true",
        help="Skip the membrane-ring DAB measurement step",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Delete existing cell data and re-run from scratch",
    )
    parser.add_argument(
        "--membrane-only",
        action="store_true",
        help="Only run the membrane measurement step (assumes pipelines already ran)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    t_start = time.time()

    print("=" * 60)
    print("  PIPELINE COMPARISON: BC Control Slides")
    print("  Baseline (brightfield_nuclei) vs Ours (brightfield_cells_nuclei)")
    print("=" * 60)
    print()

    # ── Step 0: Validate inputs ──

    if not PIPELINE_SCRIPT.exists():
        logger.error("Pipeline script not found: %s", PIPELINE_SCRIPT)
        sys.exit(1)

    if not OUR_MODEL_PT.exists():
        logger.error("Our model not found: %s", OUR_MODEL_PT)
        sys.exit(1)

    for slide_id, slide_path in SLIDE_PATHS.items():
        if not slide_path.exists():
            logger.error("Slide not found: %s", slide_path)
            sys.exit(1)

    # ── Step 1: Setup project directories ──

    logger.info("Step 1: Setting up project directories")
    setup_project_dir(V1_PROJECT)
    setup_project_dir(V2_PROJECT)

    if args.force_rerun:
        for project_dir in [V1_PROJECT, V2_PROJECT]:
            for slide_id in SLIDES:
                parquet = project_dir / "cell_data" / f"{slide_id}_cells.parquet"
                if parquet.exists():
                    parquet.unlink()
                    logger.info("  Deleted existing: %s", parquet)

    # ── Step 2: Install our model ──

    logger.info("Step 2: Installing our model in InstanSeg cache")
    install_our_model()

    if args.membrane_only:
        logger.info("--membrane-only mode: skipping pipeline runs")
        args.skip_baseline = True
        args.skip_v2 = True

    # ── Step 3: Run baseline pipeline ──

    if not args.skip_baseline:
        logger.info("Step 3: Running baseline pipeline (brightfield_nuclei)")
        ok = run_pipeline(
            model_name="brightfield_nuclei",
            project_dir=V1_PROJECT,
            label="v1-baseline",
        )
        if not ok:
            logger.error("Baseline pipeline failed. Continuing with comparison anyway.")
    else:
        logger.info("Step 3: SKIPPED (--skip-baseline)")

    # ── Step 4: Run our model pipeline ──

    if not args.skip_v2:
        logger.info("Step 4: Running our model pipeline (brightfield_cells_nuclei)")
        ok = run_pipeline(
            model_name="brightfield_cells_nuclei",
            project_dir=V2_PROJECT,
            label="v2-ours",
        )
        if not ok:
            logger.error("V2 pipeline failed. Continuing with comparison anyway.")
    else:
        logger.info("Step 4: SKIPPED (--skip-v2)")

    # ── Step 5: Read results ──

    logger.info("Step 5: Reading pipeline outputs")
    v1_data = read_parquet_results(V1_PROJECT)
    v2_data = read_parquet_results(V2_PROJECT)
    baseline_csv = read_baseline_csv()

    # ── Step 6: Add membrane-ring DAB columns to v2 ──

    if not args.skip_membrane and v2_data:
        logger.info("Step 6: Adding membrane-ring DAB columns to v2 output")
        add_membrane_ring_columns(V2_PROJECT, progress_label="v2")
        # Re-read v2 data with new columns
        v2_data = read_parquet_results(V2_PROJECT)
    else:
        logger.info("Step 6: SKIPPED (--skip-membrane or no v2 data)")

    # ── Step 7: Compare and report ──

    logger.info("Step 7: Computing comparison metrics")
    comparison = compare_results(v1_data, v2_data, baseline_csv)

    # Print table
    print_comparison_table(comparison)

    # ── Step 8: Save results ──

    logger.info("Step 8: Saving results")
    EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Save JSON
    output_json = EVAL_OUTPUT / "comparison_results.json"
    with open(output_json, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("  Saved comparison JSON: %s", output_json)

    # Also save a human-readable summary
    summary_txt = EVAL_OUTPUT / "comparison_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("PIPELINE COMPARISON SUMMARY\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 70}\n\n")

        for slide_id, slide_data in comparison.get("slides", {}).items():
            f.write(f"Slide: {slide_id}\n")
            f.write(f"  V1 (baseline) cells: {slide_data.get('v1_total_cells', '?'):>10,}\n")
            f.write(f"  V2 (our model) cells: {slide_data.get('v2_total_cells', '?'):>9,}\n")
            ratio = slide_data.get("cell_count_ratio", float("nan"))
            if not (ratio != ratio):
                f.write(f"  Cell count ratio (v2/v1): {ratio:.3f}\n")
            f.write("\n")

            for region, rdata in slide_data.get("regions", {}).items():
                f.write(f"  Region: {region}\n")
                f.write(f"    V1 cells: {rdata.get('v1_cells', 0):,}\n")
                f.write(f"    V2 cells: {rdata.get('v2_cells', 0):,}\n")

                csv_base = rdata.get("csv_baseline", {})
                if csv_base:
                    f.write(f"    CSV baseline H-score:     {csv_base.get('h_score', 'N/A')}\n")
                    f.write(f"    CSV baseline pct_positive: {csv_base.get('pct_positive', 'N/A')}%\n")
                    f.write(f"    CSV baseline mean_dab_od:  {csv_base.get('mean_dab_od', 'N/A')}\n")

                ring_data = rdata.get("v2_membrane_ring", {})
                if ring_data:
                    f.write(f"    V2 membrane-ring H-score:     {ring_data.get('h_score', 'N/A'):.1f}\n")
                    f.write(f"    V2 membrane-ring pct_positive: {ring_data.get('pct_positive', 'N/A'):.1f}%\n")
                    f.write(f"    V2 membrane-ring mean_dab:     {ring_data.get('mean_dab', 'N/A'):.4f}\n")
                    f.write(f"    V2 membrane-ring completeness: (per cell, see JSON)\n")
                    f.write(f"    V2 intensity breakdown:\n")
                    f.write(f"      Negative: {ring_data.get('pct_negative', 0):.1f}%\n")
                    f.write(f"      1+:       {ring_data.get('pct_1plus', 0):.1f}%\n")
                    f.write(f"      2+:       {ring_data.get('pct_2plus', 0):.1f}%\n")
                    f.write(f"      3+:       {ring_data.get('pct_3plus', 0):.1f}%\n")
                f.write("\n")

    logger.info("  Saved summary text: %s", summary_txt)

    elapsed_total = time.time() - t_start
    print()
    print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"  Results: {output_json}")
    print(f"  Summary: {summary_txt}")
    print()


if __name__ == "__main__":
    main()
