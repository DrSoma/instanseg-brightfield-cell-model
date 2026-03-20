#!/usr/bin/env python3
"""Per-cell grade heatmap overlays for Orion viewer.

Generates TWO overlay types per slide:
1. DISCRETE: Each cell dot colored by grade (gray=0, blue=1+, yellow=2+, red=3+)
2. CONTINUOUS: Each cell dot colored by raw membrane DAB OD (cool→hot gradient)

The continuous version is the most informative — it shows the actual measurement
without artificial binning, making the 1+/2+ and 2+/3+ boundary zones visible.

Output: RGBA PNG overlays at 8192px width for Orion slide viewer.

Usage:
    python scripts/14_percell_grade_heatmap.py
    python scripts/14_percell_grade_heatmap.py --cohort-dir /media/fernandosoto/DATA/cohort_v1/cell_data
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "percell_heatmaps"

OUTPUT_WIDTH = 8192
CELL_DOT_RADIUS = 4
OVERLAY_ALPHA_GRADE = 0.7    # Stronger alpha for discrete grades
OVERLAY_ALPHA_CONTINUOUS = 0.6
TISSUE_THRESHOLD = 220

# Grade colors (RGBA 0-255)
GRADE_COLORS = {
    0: (160, 160, 160, int(0.5 * 255)),   # Gray — negative
    1: (0, 128, 255, int(0.7 * 255)),     # Blue — 1+
    2: (255, 220, 0, int(0.8 * 255)),     # Yellow — 2+
    3: (255, 0, 0, int(0.9 * 255)),       # Red — 3+
}

# Continuous colormap: cool (low DAB) → hot (high DAB)
# Using a custom diverging map: blue → white → yellow → red
CONTINUOUS_CMAP = plt.cm.RdYlBu_r  # Red-Yellow-Blue reversed = blue(low) → red(high)

# Slide directories
SLIDE_DIRS = [
    Path("/media/fernandosoto/DATA/CLDN18 slides"),
    Path("/tmp/bc_slides"),
]
SLIDE_EXTS = {".ndpi", ".svs", ".tiff", ".tif"}


def find_slide(name: str) -> Path | None:
    for d in SLIDE_DIRS:
        if not d.exists():
            continue
        for ext in SLIDE_EXTS:
            p = d / f"{name}{ext}"
            if p.exists():
                return p
    return None


def generate_heatmaps(slide_name: str, parquet_path: Path, output_dir: Path) -> bool:
    """Generate both discrete and continuous per-cell heatmaps for one slide."""
    import openslide

    slide_path = find_slide(slide_name)
    if slide_path is None:
        return False

    # Read parquet — only need centroids + membrane columns
    schema = pq.read_schema(parquet_path)
    if "membrane_ring_dab" not in schema.names:
        return False

    cols = ["centroid_x", "centroid_y", "membrane_ring_dab", "cldn18_composite_grade",
            "membrane_completeness"]
    table = pq.read_table(parquet_path, columns=[c for c in cols if c in schema.names])

    cx = table.column("centroid_x").to_numpy()
    cy = table.column("centroid_y").to_numpy()
    dab = table.column("membrane_ring_dab").to_numpy()
    grades = table.column("cldn18_composite_grade").to_numpy().astype(int)

    n_cells = len(cx)
    if n_cells == 0:
        return False

    # Get slide dimensions for scaling
    slide = openslide.OpenSlide(str(slide_path))
    w0, h0 = slide.dimensions
    target_h = int(h0 * (OUTPUT_WIDTH / w0))
    thumb = slide.get_thumbnail((OUTPUT_WIDTH, target_h))
    thumb_np = np.array(thumb.convert("RGB"))
    slide.close()

    actual_h, actual_w = thumb_np.shape[:2]
    scale = actual_w / w0

    # Scale centroids
    cx_s = (cx * scale).astype(np.int32)
    cy_s = (cy * scale).astype(np.int32)
    valid = (cx_s >= 0) & (cx_s < actual_w) & (cy_s >= 0) & (cy_s < actual_h)

    # Tissue mask
    gray = np.mean(thumb_np, axis=2)
    tissue_mask = gray < TISSUE_THRESHOLD

    # ── 1. DISCRETE GRADE HEATMAP ──
    grade_rgba = np.zeros((actual_h, actual_w, 4), dtype=np.uint8)

    for i in np.where(valid)[0]:
        g = grades[i]
        if g < 0 or g > 3:
            g = 0
        color = GRADE_COLORS[g]
        cv2.circle(grade_rgba, (cx_s[i], cy_s[i]), CELL_DOT_RADIUS, color, -1)

    # Zero alpha outside tissue
    grade_rgba[~tissue_mask, 3] = 0

    out_grade = output_dir / f"{slide_name}_grade_discrete.png"
    Image.fromarray(grade_rgba, "RGBA").save(str(out_grade))

    # ── 2. CONTINUOUS DAB HEATMAP ──
    # Normalize DAB to 0-1 range for colormap
    valid_dab = dab[valid & np.isfinite(dab)]
    if len(valid_dab) == 0:
        return True  # Only discrete was generated

    dab_lo = np.percentile(valid_dab, 2)
    dab_hi = np.percentile(valid_dab, 98)
    if dab_hi <= dab_lo:
        dab_hi = dab_lo + 0.01

    cont_rgba = np.zeros((actual_h, actual_w, 4), dtype=np.uint8)

    for i in np.where(valid)[0]:
        d = dab[i]
        if not np.isfinite(d):
            continue
        # Normalize to 0-1
        d_norm = np.clip((d - dab_lo) / (dab_hi - dab_lo), 0, 1)
        # Map through colormap
        r, g, b, _ = CONTINUOUS_CMAP(d_norm)
        alpha = int(OVERLAY_ALPHA_CONTINUOUS * 255)
        color = (int(r * 255), int(g * 255), int(b * 255), alpha)
        cv2.circle(cont_rgba, (cx_s[i], cy_s[i]), CELL_DOT_RADIUS, color, -1)

    cont_rgba[~tissue_mask, 3] = 0

    out_cont = output_dir / f"{slide_name}_dab_continuous.png"
    Image.fromarray(cont_rgba, "RGBA").save(str(out_cont))

    # ── 3. GRADE DISTRIBUTION SUMMARY (small text overlay) ──
    n0 = (grades[valid] == 0).sum()
    n1 = (grades[valid] == 1).sum()
    n2 = (grades[valid] == 2).sum()
    n3 = (grades[valid] == 3).sum()
    n_valid = valid.sum()

    return True


def main():
    parser = argparse.ArgumentParser(description="Per-cell grade heatmaps")
    parser.add_argument("--cohort-dir", type=Path,
                        default=Path("/media/fernandosoto/DATA/cohort_v1/cell_data"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-slides", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Per-Cell Grade Heatmap Generator")
    print("  Discrete (0/1+/2+/3+) + Continuous (DAB OD gradient)")
    print("=" * 60)

    # Find slides with membrane columns
    parquets = sorted(args.cohort_dir.glob("*_cells.parquet"))
    ready = []
    for p in parquets:
        schema = pq.read_schema(p)
        if "membrane_ring_dab" in schema.names:
            name = p.stem.replace("_cells", "")
            # Skip if already generated
            out_grade = args.output_dir / f"{name}_grade_discrete.png"
            if not out_grade.exists():
                ready.append(p)

    if args.max_slides > 0:
        ready = ready[:args.max_slides]

    print(f"  Slides ready: {len(ready)} (with membrane columns, not yet generated)")
    print()

    t0 = time.time()
    success = 0
    failed = 0

    for i, pq_path in enumerate(ready):
        name = pq_path.stem.replace("_cells", "")
        t1 = time.time()
        print(f"  [{i+1}/{len(ready)}] {name}...", end=" ", flush=True)

        if generate_heatmaps(name, pq_path, args.output_dir):
            elapsed = time.time() - t1
            # Quick stats
            schema = pq.read_schema(pq_path)
            table = pq.read_table(pq_path, columns=["cldn18_composite_grade"])
            grades = table.column("cldn18_composite_grade").to_numpy()
            n = len(grades)
            pct3 = (grades == 3).sum() / n * 100 if n > 0 else 0
            print(f"OK ({n:,} cells, {pct3:.0f}% 3+, {elapsed:.1f}s)")
            success += 1
        else:
            print("SKIP")
            failed += 1

    total = time.time() - t0
    print(f"\n  Done: {success} generated, {failed} skipped in {total:.0f}s")
    print(f"  Output: {args.output_dir}")

    # Generate legend image
    _generate_legend(args.output_dir)


def _generate_legend(output_dir: Path):
    """Generate a color legend image."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # Discrete legend
    ax = axes[0]
    labels = ["Negative (0)", "Weak (1+)", "Moderate (2+)", "Strong (3+)"]
    colors = [(0.63, 0.63, 0.63), (0, 0.5, 1), (1, 0.86, 0), (1, 0, 0)]
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.barh(i, 1, color=color, edgecolor="black", linewidth=0.5)
        ax.text(0.5, i, label, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 3.5)
    ax.invert_yaxis()
    ax.set_title("Discrete Grade", fontsize=13, fontweight="bold")
    ax.axis("off")

    # Continuous legend
    ax = axes[1]
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap=CONTINUOUS_CMAP, extent=[0, 1, 0, 1])
    ax.set_xlabel("Membrane DAB Optical Density", fontsize=11)
    ax.set_title("Continuous DAB Intensity", fontsize=13, fontweight="bold")
    ax.set_yticks([])

    plt.suptitle("CLDN18.2 Per-Cell Heatmap Legend\n(THRESHOLDS UNCALIBRATED — Research Use Only)",
                 fontsize=12, color="red")
    plt.tight_layout()
    plt.savefig(output_dir / "legend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Legend saved: {output_dir / 'legend.png'}")


if __name__ == "__main__":
    main()
