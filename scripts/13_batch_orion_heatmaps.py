#!/usr/bin/env python3
"""E1: Generate Orion heatmaps for all processed cohort slides.

Reads parquet files from cohort output and generates RGBA PNG heatmaps
at 8192px width with 4px cell dots, sigma=3.0 smoothing, gap_fill=12.

Outputs to evaluation/orion_heatmaps/ with one heatmap per slide per metric:
  - {slide}_composite_grade.png (hot colormap)
  - {slide}_cell_density.png (viridis colormap)

Usage:
    python scripts/13_batch_orion_heatmaps.py [--input-dir /tmp/cohort_v1/cell_data]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "orion_heatmaps"

# Configuration
OUTPUT_WIDTH = 8192
GAUSSIAN_SIGMA = 3.0
GAP_FILL_SIZE = 12
CELL_DOT_RADIUS = 4
OVERLAY_ALPHA = 0.55
TISSUE_THRESHOLD = 220
PERCENTILE_LO = 2
PERCENTILE_HI = 98
MIN_DISPLAY_THRESHOLD = 0.005

# Slide directories to search
SLIDE_DIRS = [
    Path("/media/fernandosoto/DATA/CLDN18 slides"),
    Path("/tmp/bc_slides"),
    Path("/pathodata/Claudin18_project"),
]
SLIDE_EXTS = {".ndpi", ".svs", ".tiff", ".mrxs"}


def find_slide(slide_name: str) -> Path | None:
    """Search for a slide file across known directories."""
    for sdir in SLIDE_DIRS:
        if not sdir.exists():
            continue
        for ext in SLIDE_EXTS:
            candidate = sdir / f"{slide_name}{ext}"
            if candidate.exists():
                return candidate
        # Also search subdirectories one level deep
        for sub in sdir.iterdir():
            if sub.is_dir():
                for ext in SLIDE_EXTS:
                    candidate = sub / f"{slide_name}{ext}"
                    if candidate.exists():
                        return candidate
    return None


def generate_heatmap(slide_path: Path, parquet_path: Path, slide_name: str,
                     output_dir: Path) -> bool:
    """Generate heatmap overlays for a single slide."""
    import openslide

    try:
        # Load slide thumbnail
        slide = openslide.OpenSlide(str(slide_path))
        w0, h0 = slide.dimensions
        target_height = int(h0 * (OUTPUT_WIDTH / w0))
        thumb = slide.get_thumbnail((OUTPUT_WIDTH, target_height))
        thumb_np = np.array(thumb.convert("RGB"))
        slide.close()

        actual_h, actual_w = thumb_np.shape[:2]
        scale = actual_w / w0

        # Tissue mask
        gray = np.mean(thumb_np[:, :, :3], axis=2)
        tissue_mask = gray < TISSUE_THRESHOLD

        # Load cell data
        table = pq.read_table(parquet_path)
        df_dict = {col: table.column(col).to_numpy() for col in ["centroid_x", "centroid_y"]
                   if col in table.column_names}

        if "centroid_x" not in df_dict or "centroid_y" not in df_dict:
            print(f"    WARNING: Missing centroid columns in {parquet_path.name}")
            return False

        cx = df_dict["centroid_x"]
        cy = df_dict["centroid_y"]

        # Check for membrane columns
        has_dab = "membrane_ring_dab" in table.column_names
        has_grade = "cldn18_composite_grade" in table.column_names

        # Scale centroids to thumbnail coordinates
        cx_t = (cx * scale).astype(np.int32)
        cy_t = (cy * scale).astype(np.int32)

        # Clip to bounds
        valid = (cx_t >= 0) & (cx_t < actual_w) & (cy_t >= 0) & (cy_t < actual_h)
        cx_t = cx_t[valid]
        cy_t = cy_t[valid]
        n_valid = len(cx_t)

        # --- Generate density heatmap ---
        density = np.zeros((actual_h, actual_w), dtype=np.float32)
        for x, y in zip(cx_t, cy_t):
            cv2.circle(density, (x, y), CELL_DOT_RADIUS, 1.0, -1)

        # Smooth
        density = gaussian_filter(density, sigma=GAUSSIAN_SIGMA)

        # Gap filling
        if GAP_FILL_SIZE > 0:
            density = maximum_filter(density, size=GAP_FILL_SIZE)
            density = gaussian_filter(density, sigma=GAUSSIAN_SIGMA * 0.5)

        # Mask to tissue
        density[~tissue_mask] = 0

        # Normalize
        vals = density[density > MIN_DISPLAY_THRESHOLD]
        if len(vals) > 0:
            lo = np.percentile(vals, PERCENTILE_LO)
            hi = np.percentile(vals, PERCENTILE_HI)
            if hi > lo:
                density = np.clip((density - lo) / (hi - lo), 0, 1)

        # Apply colormap
        cmap = plt.cm.viridis
        density_rgba = (cmap(density) * 255).astype(np.uint8)
        density_rgba[density < MIN_DISPLAY_THRESHOLD, 3] = 0
        density_rgba[density >= MIN_DISPLAY_THRESHOLD, 3] = int(OVERLAY_ALPHA * 255)

        # Save density heatmap
        out_path = output_dir / f"{slide_name}_density.png"
        Image.fromarray(density_rgba, "RGBA").save(str(out_path))

        # --- Generate DAB/grade heatmap (if membrane columns exist) ---
        if has_dab:
            dab_values = table.column("membrane_ring_dab").to_numpy()[valid]

            dab_map = np.zeros((actual_h, actual_w), dtype=np.float32)
            for x, y, d in zip(cx_t, cy_t, dab_values):
                if np.isfinite(d) and d > 0:
                    cv2.circle(dab_map, (x, y), CELL_DOT_RADIUS, float(d), -1)

            dab_map = gaussian_filter(dab_map, sigma=GAUSSIAN_SIGMA)
            if GAP_FILL_SIZE > 0:
                dab_map = maximum_filter(dab_map, size=GAP_FILL_SIZE)
                dab_map = gaussian_filter(dab_map, sigma=GAUSSIAN_SIGMA * 0.5)

            dab_map[~tissue_mask] = 0

            vals = dab_map[dab_map > MIN_DISPLAY_THRESHOLD]
            if len(vals) > 0:
                lo = np.percentile(vals, PERCENTILE_LO)
                hi = np.percentile(vals, PERCENTILE_HI)
                if hi > lo:
                    dab_map = np.clip((dab_map - lo) / (hi - lo), 0, 1)

            cmap_hot = plt.cm.hot
            dab_rgba = (cmap_hot(dab_map) * 255).astype(np.uint8)
            dab_rgba[dab_map < MIN_DISPLAY_THRESHOLD, 3] = 0
            dab_rgba[dab_map >= MIN_DISPLAY_THRESHOLD, 3] = int(OVERLAY_ALPHA * 255)

            out_path = output_dir / f"{slide_name}_dab_intensity.png"
            Image.fromarray(dab_rgba, "RGBA").save(str(out_path))

        return True

    except Exception as e:
        print(f"    ERROR processing {slide_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="E1: Batch Orion heatmaps")
    parser.add_argument("--input-dir", type=Path,
                        default=Path("/tmp/cohort_v1/cell_data"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-slides", type=int, default=0,
                        help="Process only first N slides (0=all)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  E1: Batch Orion Heatmap Generation")
    print("=" * 60)
    print(f"  Input: {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Resolution: {OUTPUT_WIDTH}px width")

    # Find all parquet files
    parquet_files = sorted(args.input_dir.glob("*_cells.parquet"))
    if args.max_slides > 0:
        parquet_files = parquet_files[:args.max_slides]

    print(f"  Slides to process: {len(parquet_files)}")
    print()

    success = 0
    failed = 0
    skipped = 0
    t0 = time.time()

    for i, pq_path in enumerate(parquet_files):
        slide_name = pq_path.stem.replace("_cells", "")

        # Check if already generated
        out_density = args.output_dir / f"{slide_name}_density.png"
        if out_density.exists():
            skipped += 1
            continue

        # Find slide
        slide_path = find_slide(slide_name)
        if slide_path is None:
            print(f"  [{i+1}/{len(parquet_files)}] {slide_name}: SLIDE NOT FOUND")
            failed += 1
            continue

        print(f"  [{i+1}/{len(parquet_files)}] {slide_name}...", end=" ", flush=True)
        t1 = time.time()

        if generate_heatmap(slide_path, pq_path, slide_name, args.output_dir):
            elapsed = time.time() - t1
            print(f"OK ({elapsed:.1f}s)")
            success += 1
        else:
            failed += 1

    elapsed_total = time.time() - t0
    print(f"\n  Done in {elapsed_total:.0f}s")
    print(f"  Success: {success}, Failed: {failed}, Skipped: {skipped}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
