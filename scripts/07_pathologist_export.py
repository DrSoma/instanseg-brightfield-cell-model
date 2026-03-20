#!/usr/bin/env python3
"""Export 30 stratified tiles with cell boundary overlays for pathologist review.

Selects tiles stratified by composite grade (10 low, 10 mid, 10 high) from
BC_ClassII and BC_ClassIII slides, draws color-coded cell boundaries, and
generates per-tile CSVs plus an HTML summary page for Dr. Fiset.

Usage:
    python scripts/07_pathologist_export.py [--output-dir DIR] [--n-tiles N]
"""

from __future__ import annotations

import argparse
import html as html_mod
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure offscreen rendering (must be set before any Qt/matplotlib imports)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely import from_wkb

try:
    import openslide
except ImportError:
    print("ERROR: openslide-python is required. Install with: pip install openslide-python")
    sys.exit(1)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Slide paths
SLIDES_DIR = Path("/tmp/bc_slides")
SLIDE_IDS = ["BC_ClassII", "BC_ClassIII"]
SLIDE_PATHS = {s: SLIDES_DIR / f"{s}.ndpi" for s in SLIDE_IDS}

# Parquet data
CELL_DATA_DIR = Path("/tmp/pipeline_comparison/v2_cells_nuclei/cell_data")
PARQUET_PATHS = {s: CELL_DATA_DIR / f"{s}_cells.parquet" for s in SLIDE_IDS}

# Pipeline tile parameters
TILE_SIZE = 512       # pixels at reading level
TILE_PADDING = 80     # padding on each side
READING_LEVEL = 1     # level 1 = 2x downsample, ~0.46 um/px for these slides

# Grade color map: grade -> (R, G, B)
GRADE_COLORS = {
    0: (128, 128, 128),  # gray — negative
    1: (0, 128, 255),    # blue — 1+
    2: (255, 255, 0),    # yellow — 2+
    3: (255, 0, 0),      # red — 3+
}

GRADE_LABELS = {
    0: "Negative (0)",
    1: "Weak (1+)",
    2: "Moderate (2+)",
    3: "Strong (3+)",
}

CATEGORY_NAMES = {
    "low": "Low grade (expected 0/1+)",
    "mid": "Mid grade (expected 2+)",
    "high": "High grade (expected 3+)",
}


# ── Tile selection ───────────────────────────────────────────────────────────


def compute_tile_stats(
    df: pd.DataFrame, slide_id: str, downsample: float
) -> pd.DataFrame:
    """Group cells by tile and compute per-tile statistics.

    Args:
        df: Cell dataframe with centroid_x, centroid_y, cldn18_composite_grade.
        slide_id: Slide identifier string.
        downsample: Level downsample factor (e.g. 2.0 for level 1).

    Returns:
        DataFrame with columns: tile_x, tile_y, n_cells, mean_grade,
        grade_0_pct, grade_1_pct, grade_2_pct, grade_3_pct, slide_id.
    """
    tile_step = (TILE_SIZE - 2 * TILE_PADDING) * downsample
    df = df.copy()
    df["tile_x"] = (df["centroid_x"] // tile_step).astype(int)
    df["tile_y"] = (df["centroid_y"] // tile_step).astype(int)

    stats = df.groupby(["tile_x", "tile_y"]).agg(
        n_cells=("cell_id", "count"),
        mean_grade=("cldn18_composite_grade", "mean"),
        grade_0_pct=("cldn18_composite_grade", lambda x: (x == 0).mean() * 100),
        grade_1_pct=("cldn18_composite_grade", lambda x: (x == 1).mean() * 100),
        grade_2_pct=("cldn18_composite_grade", lambda x: (x == 2).mean() * 100),
        grade_3_pct=("cldn18_composite_grade", lambda x: (x == 3).mean() * 100),
    ).reset_index()

    stats["slide_id"] = slide_id
    stats["tile_step"] = tile_step
    return stats


def select_stratified_tiles(
    all_tile_stats: pd.DataFrame, n_per_category: int = 10, min_cells: int = 10
) -> pd.DataFrame:
    """Select tiles stratified by grade into low/mid/high categories.

    Filters out tiles with too few cells, then selects n_per_category tiles
    from each tercile of mean_grade, mixing slides roughly equally.

    Args:
        all_tile_stats: Combined tile stats from all slides.
        n_per_category: Number of tiles to select per category.
        min_cells: Minimum cells required in a tile.

    Returns:
        DataFrame of selected tiles with an added 'category' column.
    """
    # Filter tiles with enough cells
    valid = all_tile_stats[all_tile_stats["n_cells"] >= min_cells].copy()
    if len(valid) == 0:
        raise ValueError("No tiles with enough cells found")

    logger.info(
        "  Tiles with >= %d cells: %d / %d",
        min_cells, len(valid), len(all_tile_stats),
    )

    # Compute tercile boundaries
    p33 = np.percentile(valid["mean_grade"], 33.3)
    p67 = np.percentile(valid["mean_grade"], 66.7)
    logger.info("  Grade terciles: p33=%.2f, p67=%.2f", p33, p67)

    # Assign categories
    valid["category"] = "mid"
    valid.loc[valid["mean_grade"] <= p33, "category"] = "low"
    valid.loc[valid["mean_grade"] >= p67, "category"] = "high"

    selected_parts = []
    for cat in ["low", "mid", "high"]:
        pool = valid[valid["category"] == cat]
        if len(pool) == 0:
            logger.warning("  No tiles in category '%s'", cat)
            continue

        # Try to get roughly equal numbers from each slide
        slides = pool["slide_id"].unique()
        per_slide = max(1, n_per_category // len(slides))
        remainder = n_per_category - per_slide * len(slides)

        cat_selected = []
        for i, sid in enumerate(slides):
            slide_pool = pool[pool["slide_id"] == sid]
            n_take = per_slide + (1 if i < remainder else 0)
            n_take = min(n_take, len(slide_pool))
            # Sample spread across the grade range within this category
            sampled = slide_pool.sort_values("mean_grade").iloc[
                np.linspace(0, len(slide_pool) - 1, n_take, dtype=int)
            ]
            cat_selected.append(sampled)

        cat_df = pd.concat(cat_selected, ignore_index=True)

        # If we still need more tiles, fill from any slide
        if len(cat_df) < n_per_category:
            remaining_pool = pool[
                ~pool.index.isin(cat_df.index)
            ]
            extra = remaining_pool.sample(
                n=min(n_per_category - len(cat_df), len(remaining_pool)),
                random_state=42,
            )
            cat_df = pd.concat([cat_df, extra], ignore_index=True)

        cat_df = cat_df.head(n_per_category)
        logger.info(
            "  Category '%s': selected %d tiles (mean grade range: %.2f - %.2f)",
            cat, len(cat_df), cat_df["mean_grade"].min(), cat_df["mean_grade"].max(),
        )
        selected_parts.append(cat_df)

    selected = pd.concat(selected_parts, ignore_index=True)
    return selected


# ── Tile rendering ───────────────────────────────────────────────────────────


def extract_tile_image(
    slide: openslide.OpenSlide,
    tile_x: int,
    tile_y: int,
    tile_step: float,
    level: int,
    downsample: float,
) -> np.ndarray | None:
    """Read a tile from the slide at the given grid position.

    Args:
        slide: Open OpenSlide object.
        tile_x: Tile grid x index.
        tile_y: Tile grid y index.
        tile_step: Step size in level-0 pixels.
        level: Reading level.
        downsample: Downsample factor for the reading level.

    Returns:
        RGB numpy array of shape (TILE_SIZE, TILE_SIZE, 3), or None on error.
    """
    # Top-left corner in level-0 coordinates
    x0_l0 = int(tile_x * tile_step)
    y0_l0 = int(tile_y * tile_step)

    # Clamp to slide bounds
    slide_w, slide_h = slide.dimensions
    if x0_l0 < 0 or y0_l0 < 0 or x0_l0 >= slide_w or y0_l0 >= slide_h:
        return None

    try:
        region = slide.read_region((x0_l0, y0_l0), level, (TILE_SIZE, TILE_SIZE))
        return np.array(region.convert("RGB"))
    except Exception as e:
        logger.warning("  Failed to read tile (%d, %d): %s", tile_x, tile_y, e)
        return None


def draw_cell_boundaries(
    img: np.ndarray,
    cells_df: pd.DataFrame,
    x0_l0: float,
    y0_l0: float,
    downsample: float,
    line_width: int = 2,
) -> np.ndarray:
    """Draw color-coded cell boundary outlines on a tile image.

    Args:
        img: RGB tile image (H, W, 3) uint8.
        cells_df: DataFrame of cells in this tile, must have polygon_wkb
                  and cldn18_composite_grade columns.
        x0_l0: Tile origin x in level-0 coordinates.
        y0_l0: Tile origin y in level-0 coordinates.
        downsample: Level downsample factor.
        line_width: Boundary line width in pixels.

    Returns:
        Annotated RGB image.
    """
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    n_drawn = 0
    for _, row in cells_df.iterrows():
        wkb_data = row.get("polygon_wkb")
        if wkb_data is None:
            continue

        try:
            poly = from_wkb(wkb_data)
        except Exception:
            continue

        if poly.is_empty:
            continue

        grade = int(row.get("cldn18_composite_grade", 0))
        color = GRADE_COLORS.get(grade, (128, 128, 128))

        # Transform exterior coordinates from level-0 to tile-local pixels
        coords = np.array(poly.exterior.coords)
        local_x = (coords[:, 0] - x0_l0) / downsample
        local_y = (coords[:, 1] - y0_l0) / downsample

        # Build point list for PIL
        points = list(zip(local_x.tolist(), local_y.tolist()))
        if len(points) < 3:
            continue

        draw.polygon(points, outline=color, width=line_width)
        n_drawn += 1

    return np.array(pil_img)


# ── Per-tile CSV export ──────────────────────────────────────────────────────


def export_tile_csv(cells_df: pd.DataFrame, output_path: Path) -> None:
    """Export per-cell data for a tile to CSV.

    Args:
        cells_df: DataFrame of cells in this tile.
        output_path: Path to write the CSV file.
    """
    export_cols = {
        "cell_id": "cell_id",
        "centroid_x": "centroid_x",
        "centroid_y": "centroid_y",
        "membrane_thickness_px": "thickness_px",
        "membrane_thickness_um": "thickness_um",
        "membrane_completeness": "completeness",
        "membrane_ring_dab": "dab_intensity",
        "cldn18_composite_grade": "composite_grade",
    }

    out = pd.DataFrame()
    for src_col, dst_col in export_cols.items():
        if src_col in cells_df.columns:
            out[dst_col] = cells_df[src_col].values
        else:
            out[dst_col] = np.nan

    out.to_csv(output_path, index=False, float_format="%.4f")


# ── HTML summary ─────────────────────────────────────────────────────────────


def generate_summary_html(
    tile_records: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Generate an HTML overview page with thumbnail grid and statistics.

    Args:
        tile_records: List of dicts with keys: tile_idx, slide_id, category,
            mean_grade, n_cells, image_filename, csv_filename, grade_counts.
        output_dir: Output directory where images/CSVs are already saved.

    Returns:
        Path to the generated summary.html file.
    """
    html_path = output_dir / "summary.html"

    # Compute per-category statistics
    cat_stats: dict[str, dict[str, Any]] = {}
    for cat in ["low", "mid", "high"]:
        cat_tiles = [r for r in tile_records if r["category"] == cat]
        if not cat_tiles:
            continue
        total_cells = sum(r["n_cells"] for r in cat_tiles)
        mean_grades = [r["mean_grade"] for r in cat_tiles]
        grade_totals = {g: 0 for g in range(4)}
        for r in cat_tiles:
            for g, c in r["grade_counts"].items():
                grade_totals[g] += c

        cat_stats[cat] = {
            "n_tiles": len(cat_tiles),
            "total_cells": total_cells,
            "mean_grade_range": f"{min(mean_grades):.2f} - {max(mean_grades):.2f}",
            "mean_grade_avg": f"{np.mean(mean_grades):.2f}",
            "grade_totals": grade_totals,
            "slides": list(set(r["slide_id"] for r in cat_tiles)),
        }

    # Build HTML
    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CLDN18.2 Pathologist Export — 30-Tile Review</title>
<style>
  body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
         margin: 20px; background: #f5f5f5; color: #333; }
  h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
  h2 { color: #2c3e50; margin-top: 30px; }
  h3 { color: #555; }
  .legend { display: flex; gap: 20px; margin: 15px 0; padding: 10px;
            background: white; border-radius: 8px; border: 1px solid #ddd; }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .legend-swatch { width: 20px; height: 20px; border-radius: 3px;
                   border: 1px solid #999; }
  .instructions { background: #e8f4fd; border-left: 4px solid #3498db;
                  padding: 15px 20px; margin: 20px 0; border-radius: 4px; }
  .stats-table { border-collapse: collapse; margin: 10px 0; }
  .stats-table th, .stats-table td { border: 1px solid #ccc; padding: 8px 12px;
                                      text-align: center; }
  .stats-table th { background: #3498db; color: white; }
  .stats-table tr:nth-child(even) { background: #f0f0f0; }
  .category-section { margin: 20px 0; padding: 15px; background: white;
                      border-radius: 8px; border: 1px solid #ddd; }
  .tile-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
               gap: 15px; margin: 15px 0; }
  .tile-card { background: white; border-radius: 8px; border: 1px solid #ddd;
               overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  .tile-card img { width: 100%; height: auto; display: block; }
  .tile-card .info { padding: 10px; font-size: 0.9em; }
  .tile-card .info strong { color: #2c3e50; }
  .cat-low { border-left: 4px solid #808080; }
  .cat-mid { border-left: 4px solid #ffc107; }
  .cat-high { border-left: 4px solid #e74c3c; }
  .footer { margin-top: 30px; padding: 15px; background: #ecf0f1;
            border-radius: 8px; font-size: 0.85em; color: #666; }
</style>
</head>
<body>
""")

    parts.append("<h1>CLDN18.2 Composite Grade — 30-Tile Pathologist Review</h1>\n")
    parts.append(f"<p>Generated: {time.strftime('%Y-%m-%d %H:%M')}</p>\n")

    # Instructions
    parts.append("""
<div class="instructions">
<h3>Review Instructions for Dr. Fiset</h3>
<ol>
  <li><strong>Each tile</strong> shows a 512x512 region at ~0.46 um/px from a BC control slide.</li>
  <li><strong>Cell boundaries</strong> are drawn as colored outlines based on the algorithm's
      composite grade assignment (thickness + completeness + intensity).</li>
  <li><strong>Color code:</strong> see legend below.</li>
  <li>For each tile, a <strong>CSV file</strong> is provided with per-cell measurements
      (membrane thickness, completeness, DAB intensity, composite grade).</li>
  <li>Tiles are grouped into <strong>3 categories</strong> (10 each) by mean composite grade:
      <em>Low</em> (expected negative/1+), <em>Mid</em> (expected 2+), <em>High</em> (expected 3+).</li>
  <li>Please assess whether the <strong>grade assignments match your clinical judgment</strong>
      for each tile. Note any disagreements or boundary cases.</li>
</ol>
</div>
""")

    # Color legend
    parts.append('<div class="legend">\n')
    for grade, color in GRADE_COLORS.items():
        r, g, b = color
        label = GRADE_LABELS[grade]
        parts.append(
            f'  <div class="legend-item">'
            f'<div class="legend-swatch" style="background:rgb({r},{g},{b})"></div>'
            f'<span>{html_mod.escape(label)}</span></div>\n'
        )
    parts.append("</div>\n")

    # Per-category statistics table
    parts.append("<h2>Category Statistics</h2>\n")
    parts.append('<table class="stats-table">\n')
    parts.append(
        "<tr><th>Category</th><th>Tiles</th><th>Total Cells</th>"
        "<th>Mean Grade Range</th><th>Mean Grade Avg</th>"
        "<th>Neg %</th><th>1+ %</th><th>2+ %</th><th>3+ %</th>"
        "<th>Slides</th></tr>\n"
    )
    for cat in ["low", "mid", "high"]:
        if cat not in cat_stats:
            continue
        cs = cat_stats[cat]
        gt = cs["grade_totals"]
        total = cs["total_cells"]
        pcts = {g: gt[g] / total * 100 if total > 0 else 0 for g in range(4)}
        parts.append(
            f"<tr>"
            f"<td><strong>{html_mod.escape(CATEGORY_NAMES[cat])}</strong></td>"
            f"<td>{cs['n_tiles']}</td>"
            f"<td>{cs['total_cells']:,}</td>"
            f"<td>{cs['mean_grade_range']}</td>"
            f"<td>{cs['mean_grade_avg']}</td>"
            f"<td>{pcts[0]:.1f}%</td>"
            f"<td>{pcts[1]:.1f}%</td>"
            f"<td>{pcts[2]:.1f}%</td>"
            f"<td>{pcts[3]:.1f}%</td>"
            f"<td>{', '.join(cs['slides'])}</td>"
            f"</tr>\n"
        )
    parts.append("</table>\n")

    # Tile grids per category
    for cat in ["low", "mid", "high"]:
        cat_tiles = sorted(
            [r for r in tile_records if r["category"] == cat],
            key=lambda r: r["mean_grade"],
        )
        if not cat_tiles:
            continue

        css_class = f"cat-{cat}"
        parts.append(f'\n<div class="category-section {css_class}">\n')
        parts.append(f"<h2>{html_mod.escape(CATEGORY_NAMES[cat])}</h2>\n")
        parts.append('<div class="tile-grid">\n')

        for rec in cat_tiles:
            img_fn = html_mod.escape(rec["image_filename"])
            csv_fn = html_mod.escape(rec["csv_filename"])
            gc = rec["grade_counts"]
            total = rec["n_cells"]
            grade_str = ", ".join(
                f"{GRADE_LABELS[g].split()[0]}: {gc.get(g, 0)}"
                for g in range(4)
            )
            parts.append(
                f'  <div class="tile-card">\n'
                f'    <img src="{img_fn}" alt="Tile {rec["tile_idx"]}">\n'
                f'    <div class="info">\n'
                f'      <strong>Tile {rec["tile_idx"]:02d}</strong> — '
                f'{html_mod.escape(rec["slide_id"])}<br>\n'
                f'      Cells: {total} | Mean grade: {rec["mean_grade"]:.2f}<br>\n'
                f'      {html_mod.escape(grade_str)}<br>\n'
                f'      <a href="{csv_fn}">Download CSV</a>\n'
                f'    </div>\n'
                f'  </div>\n'
            )

        parts.append("</div>\n</div>\n")

    # Footer
    parts.append("""
<div class="footer">
  <p><strong>Pipeline:</strong> InstanSeg brightfield_cells_nuclei v2 +
     membrane bar-filter (8 orientations, FWHM thickness) +
     composite grading (thickness + completeness + intensity).</p>
  <p><strong>Slides:</strong> BC_ClassII, BC_ClassIII (positive controls, CLDN18.2 IHC).</p>
  <p><strong>Tile size:</strong> 512px at level 1 (~0.46 um/px), stride 352px (padding 80px).</p>
  <p><strong>Composite grade:</strong> 0 = negative, 1+ = weak, 2+ = moderate, 3+ = strong,
     based on membrane DAB intensity, thickness (FWHM), and completeness (fraction of
     membrane perimeter with signal).</p>
</div>
""")

    parts.append("</body>\n</html>\n")

    html_content = "".join(parts)
    html_path.write_text(html_content, encoding="utf-8")
    logger.info("  Saved summary HTML: %s", html_path)
    return html_path


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export 30 stratified tiles with cell overlays for pathologist review",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "pathologist_export",
        help="Output directory for exported tiles and HTML",
    )
    parser.add_argument(
        "--n-tiles",
        type=int,
        default=10,
        help="Number of tiles per category (default: 10, total = 3 * n_tiles)",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=10,
        help="Minimum cells per tile to be eligible for selection",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Cell boundary line width in pixels",
    )
    parser.add_argument(
        "--cell-data-dir",
        type=Path,
        default=CELL_DATA_DIR,
        help="Directory containing cell parquet files",
    )
    parser.add_argument(
        "--slides-dir",
        type=Path,
        default=SLIDES_DIR,
        help="Directory containing .ndpi slide files",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    t_start = time.time()

    print("=" * 60)
    print("  PATHOLOGIST EXPORT: 30-Tile Stratified Review")
    print("=" * 60)
    print()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load parquet data ──

    logger.info("Step 1: Loading cell data")
    all_cells: dict[str, pd.DataFrame] = {}
    for slide_id in SLIDE_IDS:
        pq_path = args.cell_data_dir / f"{slide_id}_cells.parquet"
        if not pq_path.exists():
            logger.warning("  Missing parquet: %s", pq_path)
            continue
        df = pd.read_parquet(pq_path)
        logger.info("  %s: %d cells, grade dist: %s",
                     slide_id, len(df),
                     df["cldn18_composite_grade"].value_counts().sort_index().to_dict())
        all_cells[slide_id] = df

    if not all_cells:
        logger.error("No cell data found. Exiting.")
        sys.exit(1)

    # ── Step 2: Determine slide downsample ──

    logger.info("Step 2: Reading slide properties")
    slide_handles: dict[str, openslide.OpenSlide] = {}
    downsample_map: dict[str, float] = {}

    for slide_id in all_cells:
        slide_path = args.slides_dir / f"{slide_id}.ndpi"
        if not slide_path.exists():
            logger.error("  Slide not found: %s", slide_path)
            sys.exit(1)

        slide = openslide.OpenSlide(str(slide_path))
        slide_handles[slide_id] = slide

        if READING_LEVEL >= slide.level_count:
            logger.error("  Level %d not available for %s (max: %d)",
                         READING_LEVEL, slide_id, slide.level_count - 1)
            sys.exit(1)

        ds = slide.level_downsamples[READING_LEVEL]
        downsample_map[slide_id] = ds
        mpp = float(slide.properties.get("openslide.mpp-x", 0.25))
        logger.info("  %s: level %d downsample=%.1f, mpp=%.4f (effective %.4f um/px)",
                     slide_id, READING_LEVEL, ds, mpp, mpp * ds)

    # ── Step 3: Compute tile statistics and select ──

    logger.info("Step 3: Computing tile statistics and selecting tiles")
    all_tile_stats_parts = []
    for slide_id, df in all_cells.items():
        ds = downsample_map[slide_id]
        stats = compute_tile_stats(df, slide_id, ds)
        logger.info("  %s: %d tiles", slide_id, len(stats))
        all_tile_stats_parts.append(stats)

    all_tile_stats = pd.concat(all_tile_stats_parts, ignore_index=True)
    selected = select_stratified_tiles(
        all_tile_stats,
        n_per_category=args.n_tiles,
        min_cells=args.min_cells,
    )

    total_selected = len(selected)
    logger.info("  Total selected tiles: %d", total_selected)

    # ── Step 4: Extract, annotate, and save tiles ──

    logger.info("Step 4: Extracting and annotating tiles")
    tile_records: list[dict[str, Any]] = []

    for idx, (_, tile_row) in enumerate(selected.iterrows()):
        slide_id = tile_row["slide_id"]
        tile_x = int(tile_row["tile_x"])
        tile_y = int(tile_row["tile_y"])
        tile_step = float(tile_row["tile_step"])
        category = tile_row["category"]
        mean_grade = float(tile_row["mean_grade"])
        ds = downsample_map[slide_id]

        slide = slide_handles[slide_id]

        # Extract tile image
        img = extract_tile_image(slide, tile_x, tile_y, tile_step, READING_LEVEL, ds)
        if img is None:
            logger.warning("  Tile %02d: failed to extract image, skipping", idx)
            continue

        # Get cells in this tile
        df = all_cells[slide_id]
        df_tile = df.copy()
        df_tile["_tile_x"] = (df_tile["centroid_x"] // tile_step).astype(int)
        df_tile["_tile_y"] = (df_tile["centroid_y"] // tile_step).astype(int)
        cells_in_tile = df_tile[
            (df_tile["_tile_x"] == tile_x) & (df_tile["_tile_y"] == tile_y)
        ]

        # Level-0 origin of this tile
        x0_l0 = tile_x * tile_step
        y0_l0 = tile_y * tile_step

        # Draw cell boundaries
        annotated = draw_cell_boundaries(
            img, cells_in_tile, x0_l0, y0_l0, ds, line_width=args.line_width
        )

        # File names
        short_slide = slide_id.replace("BC_", "")
        img_filename = f"tile_{idx:02d}_{short_slide}_{category}.png"
        csv_filename = f"tile_{idx:02d}_{short_slide}_{category}.csv"

        # Save annotated image
        img_path = output_dir / img_filename
        Image.fromarray(annotated).save(img_path, optimize=True)

        # Save per-cell CSV
        csv_path = output_dir / csv_filename
        export_tile_csv(cells_in_tile, csv_path)

        # Grade counts for this tile
        gc = cells_in_tile["cldn18_composite_grade"].value_counts().to_dict()
        grade_counts = {int(k): int(v) for k, v in gc.items()}

        tile_records.append({
            "tile_idx": idx,
            "slide_id": slide_id,
            "category": category,
            "mean_grade": mean_grade,
            "n_cells": len(cells_in_tile),
            "image_filename": img_filename,
            "csv_filename": csv_filename,
            "grade_counts": grade_counts,
        })

        logger.info(
            "  Tile %02d: %s (%s) — %d cells, mean=%.2f, grades=%s",
            idx, slide_id, category, len(cells_in_tile), mean_grade,
            grade_counts,
        )

    # ── Step 5: Close slides ──

    for slide in slide_handles.values():
        slide.close()

    # ── Step 6: Generate HTML summary ──

    logger.info("Step 5: Generating HTML summary")
    html_path = generate_summary_html(tile_records, output_dir)

    # ── Done ──

    elapsed = time.time() - t_start
    n_images = len(tile_records)
    n_csvs = len(tile_records)

    print()
    print("=" * 60)
    print(f"  EXPORT COMPLETE")
    print(f"  Output directory: {output_dir}")
    print(f"  Tiles exported:   {n_images} images + {n_csvs} CSVs")
    print(f"  HTML summary:     {html_path}")
    print(f"  Time:             {elapsed:.1f}s")
    print("=" * 60)
    print()

    # Print per-category summary
    for cat in ["low", "mid", "high"]:
        cat_tiles = [r for r in tile_records if r["category"] == cat]
        if cat_tiles:
            grades = [r["mean_grade"] for r in cat_tiles]
            cells = sum(r["n_cells"] for r in cat_tiles)
            print(
                f"  {CATEGORY_NAMES[cat]:40s}: "
                f"{len(cat_tiles)} tiles, {cells:5d} cells, "
                f"grade {min(grades):.2f}-{max(grades):.2f}"
            )
    print()


if __name__ == "__main__":
    main()
