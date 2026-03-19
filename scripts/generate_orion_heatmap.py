#!/usr/bin/env python3
"""Generate cell density + membrane DAB heatmap overlays for Orion digital pathology viewer.

Produces RGBA PNG overlays at thumbnail resolution (2560px wide) for:
  - Membrane DAB intensity (hot colormap)
  - Cell detection density (viridis colormap)
Each with a companion colorbar legend PNG.

Usage:
    python scripts/generate_orion_heatmap.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Force headless rendering before any Qt/matplotlib imports
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import numpy as np
import openslide
import pyarrow.parquet as pq
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_WIDTH = 8192              # 3.2x higher resolution than before (was 2560)
GAUSSIAN_SIGMA = 3.0             # wider smoothing for continuity (was 1.5)
GAP_FILL_SIZE = 12               # larger gap filling radius (was 5)
CELL_DOT_RADIUS = 4              # draw each cell as a filled circle (NEW)
OVERLAY_ALPHA = 0.55
TISSUE_THRESHOLD = 220           # grayscale; below = tissue
PERCENTILE_LO = 2
PERCENTILE_HI = 98
MIN_DISPLAY_THRESHOLD = 0.005    # lower threshold to show faint cells (was 0.01)

OUTPUT_DIR = Path("/home/fernandosoto/Downloads")

SLIDES: list[dict[str, str]] = [
    {
        "name": "BC_ClassII",
        "slide": "/tmp/bc_slides/BC_ClassII.ndpi",
        "parquet": "/tmp/pipeline_comparison/v2_cells_nuclei/cell_data/BC_ClassII_cells.parquet",
    },
    {
        "name": "BC_ClassIII",
        "slide": "/tmp/bc_slides/BC_ClassIII.ndpi",
        "parquet": "/tmp/pipeline_comparison/v2_cells_nuclei/cell_data/BC_ClassIII_cells.parquet",
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SlideData:
    """All data needed for heatmap generation."""
    name: str
    thumb_np: np.ndarray
    output_width: int
    output_height: int
    scale: float
    cx: np.ndarray          # centroid x (level-0 pixels)
    cy: np.ndarray          # centroid y (level-0 pixels)
    dab: np.ndarray         # membrane_ring_dab values
    tissue_mask: np.ndarray # boolean tissue mask at thumbnail resolution


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_slide_data(slide_path: str, parquet_path: str, name: str) -> SlideData:
    """Load slide thumbnail and cell data, compute tissue mask."""

    print(f"  Loading slide: {slide_path}")
    slide = openslide.OpenSlide(slide_path)
    w0, h0 = slide.dimensions

    # Request thumbnail at target width; openslide picks the actual size
    target_height = int(h0 * (OUTPUT_WIDTH / w0))
    thumb = slide.get_thumbnail((OUTPUT_WIDTH, target_height))
    thumb_np = np.array(thumb.convert("RGB"))
    slide.close()

    # Use actual thumbnail dimensions (may differ by 1px from arithmetic)
    actual_h, actual_w = thumb_np.shape[:2]
    scale = actual_w / w0
    output_height = actual_h
    output_width = actual_w

    print(f"  Level-0 dimensions: {w0} x {h0}")
    print(f"  Thumbnail dimensions: {output_width} x {output_height} (scale={scale:.6f})")

    # Tissue mask from grayscale threshold
    gray = np.mean(thumb_np[:, :, :3], axis=2)
    tissue_mask = gray < TISSUE_THRESHOLD

    print(f"  Tissue coverage: {tissue_mask.sum()}/{tissue_mask.size} "
          f"({100 * tissue_mask.sum() / tissue_mask.size:.1f}%)")

    # Cell data
    print(f"  Loading parquet: {parquet_path}")
    table = pq.read_table(
        parquet_path,
        columns=["centroid_x", "centroid_y", "membrane_ring_dab"],
    )
    cx = table.column("centroid_x").to_numpy()
    cy = table.column("centroid_y").to_numpy()
    dab = table.column("membrane_ring_dab").to_numpy()

    print(f"  Cells loaded: {len(cx):,}")
    print(f"  DAB range: [{np.nanmin(dab):.4f}, {np.nanmax(dab):.4f}] "
          f"mean={np.nanmean(dab):.4f}")

    return SlideData(
        name=name,
        thumb_np=thumb_np,
        output_width=output_width,
        output_height=output_height,
        scale=scale,
        cx=cx,
        cy=cy,
        dab=dab,
        tissue_mask=tissue_mask,
    )


# ---------------------------------------------------------------------------
# Heatmap generation core
# ---------------------------------------------------------------------------

def scatter_to_grid(
    cx: np.ndarray,
    cy: np.ndarray,
    values: np.ndarray,
    scale: float,
    width: int,
    height: int,
    mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Map cell coordinates to a thumbnail-resolution grid.

    Parameters
    ----------
    mode : str
        'mean'  -- average the values falling in each pixel (for DAB intensity).
        'count' -- count cells per pixel (for density).

    Returns
    -------
    heatmap : float32 array (height, width)
    counts  : float32 array (height, width) -- cell count per pixel
    """
    heatmap = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.float64)

    # Vectorized binning with filled circle dots for visibility
    px = (cx * scale).astype(np.int64)
    py = (cy * scale).astype(np.int64)

    # Filter in-bounds (with margin for dot radius)
    r = CELL_DOT_RADIUS
    mask = (px >= r) & (px < width - r) & (py >= r) & (py < height - r)
    px = px[mask]
    py = py[mask]
    val = values[mask]

    # Replace NaN with 0 for accumulation
    nan_mask = np.isnan(val)
    val_clean = np.where(nan_mask, 0.0, val)
    valid_flags = (~nan_mask).astype(np.float64)

    # Draw each cell as a filled circle (radius=CELL_DOT_RADIUS pixels)
    # Pre-compute circle offsets
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    circle_mask = (xx**2 + yy**2) <= r**2
    dy_offsets = yy[circle_mask]
    dx_offsets = xx[circle_mask]

    for i in range(len(px)):
        ys = py[i] + dy_offsets
        xs = px[i] + dx_offsets
        heatmap[ys, xs] += val_clean[i]
        counts[ys, xs] += valid_flags[i]

    if mode == "mean":
        valid = counts > 0
        heatmap[valid] /= counts[valid]
    elif mode == "count":
        heatmap = counts.copy()

    return heatmap.astype(np.float32), counts.astype(np.float32)


def build_heatmap(
    raw: np.ndarray,
    tissue_mask: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Normalize, gap-fill, smooth, and tissue-mask a raw heatmap.

    Returns a float32 array in [0, 1] ready for colormap application.
    """
    valid = counts > 0
    if valid.sum() == 0:
        return np.zeros_like(raw)

    # Percentile normalization (Orion-style)
    vals = raw[valid]
    lo = np.percentile(vals, PERCENTILE_LO)
    hi = np.percentile(vals, PERCENTILE_HI)
    denom = hi - lo
    if denom < 1e-8:
        denom = 1.0
    heatmap_norm = np.clip((raw - lo) / denom, 0.0, 1.0)

    # Gap filling via dilation to bridge sparse cell gaps
    heatmap_filled = maximum_filter(heatmap_norm, size=GAP_FILL_SIZE)

    # Gaussian smoothing
    heatmap_smooth = gaussian_filter(heatmap_filled, sigma=GAUSSIAN_SIGMA)

    # Zero out non-tissue
    heatmap_smooth[~tissue_mask] = 0.0

    return heatmap_smooth


def apply_colormap(
    heatmap: np.ndarray,
    tissue_mask: np.ndarray,
    cmap_name: str,
) -> np.ndarray:
    """Apply matplotlib colormap and set alpha channel.

    Returns RGBA uint8 array (H, W, 4).
    """
    cmap = plt.get_cmap(cmap_name)
    colored = (cmap(heatmap) * 255).astype(np.uint8)  # (H, W, 4)

    # Alpha: overlay on tissue where heatmap has signal
    alpha_mask = tissue_mask & (heatmap > MIN_DISPLAY_THRESHOLD)
    alpha_value = int(OVERLAY_ALPHA * 255)
    colored[:, :, 3] = np.where(alpha_mask, alpha_value, 0).astype(np.uint8)

    return colored


# ---------------------------------------------------------------------------
# Colorbar legend generation
# ---------------------------------------------------------------------------

def make_colorbar_png(
    cmap_name: str,
    vmin: float,
    vmax: float,
    label: str,
    out_path: Path,
) -> None:
    """Save a standalone colorbar legend as a small PNG."""
    fig, ax = plt.subplots(figsize=(1.2, 4), dpi=120)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax)
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(str(out_path), bbox_inches="tight", transparent=True, dpi=120)
    plt.close(fig)
    print(f"  Colorbar saved: {out_path}")


# ---------------------------------------------------------------------------
# Main per-slide pipeline
# ---------------------------------------------------------------------------

def process_slide(cfg: dict[str, str]) -> list[Path]:
    """Generate all heatmaps for one slide. Returns list of output paths."""
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    data = load_slide_data(cfg["slide"], cfg["parquet"], name)
    outputs: list[Path] = []

    # --- DAB intensity heatmap ---
    print(f"\n  [DAB intensity heatmap]")
    dab_raw, dab_counts = scatter_to_grid(
        data.cx, data.cy, data.dab, data.scale,
        data.output_width, data.output_height, mode="mean",
    )
    dab_smooth = build_heatmap(dab_raw, data.tissue_mask, dab_counts)
    dab_rgba = apply_colormap(dab_smooth, data.tissue_mask, "hot")

    dab_path = OUTPUT_DIR / f"{name}_dab_heatmap.png"
    Image.fromarray(dab_rgba, "RGBA").save(str(dab_path))
    outputs.append(dab_path)
    print(f"  Saved: {dab_path}")

    # DAB colorbar
    vals = dab_raw[dab_counts > 0]
    dab_lo = np.percentile(vals, PERCENTILE_LO)
    dab_hi = np.percentile(vals, PERCENTILE_HI)
    dab_cbar_path = OUTPUT_DIR / f"{name}_dab_colorbar.png"
    make_colorbar_png("hot", dab_lo, dab_hi, "Membrane DAB OD", dab_cbar_path)
    outputs.append(dab_cbar_path)

    # --- Cell density heatmap ---
    print(f"\n  [Cell density heatmap]")
    ones = np.ones_like(data.dab)
    density_raw, density_counts = scatter_to_grid(
        data.cx, data.cy, ones, data.scale,
        data.output_width, data.output_height, mode="count",
    )
    density_smooth = build_heatmap(density_raw, data.tissue_mask, density_counts)
    density_rgba = apply_colormap(density_smooth, data.tissue_mask, "viridis")

    density_path = OUTPUT_DIR / f"{name}_density_heatmap.png"
    Image.fromarray(density_rgba, "RGBA").save(str(density_path))
    outputs.append(density_path)
    print(f"  Saved: {density_path}")

    # Density colorbar
    dvals = density_raw[density_counts > 0]
    dens_lo = np.percentile(dvals, PERCENTILE_LO)
    dens_hi = np.percentile(dvals, PERCENTILE_HI)
    dens_cbar_path = OUTPUT_DIR / f"{name}_density_colorbar.png"
    make_colorbar_png("viridis", dens_lo, dens_hi, "Cell density (cells/px)", dens_cbar_path)
    outputs.append(dens_cbar_path)

    return outputs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(paths: list[Path]) -> None:
    """Print file sizes and image dimensions for all outputs."""
    print(f"\n{'='*60}")
    print("OUTPUT REPORT")
    print(f"{'='*60}")
    print(f"{'File':<50} {'Size':>10} {'Dimensions':>16}")
    print("-" * 78)
    for p in paths:
        size_bytes = p.stat().st_size
        if size_bytes > 1_048_576:
            size_str = f"{size_bytes / 1_048_576:.1f} MB"
        else:
            size_str = f"{size_bytes / 1024:.0f} KB"
        img = Image.open(str(p))
        dim_str = f"{img.width} x {img.height}"
        img.close()
        print(f"  {p.name:<48} {size_str:>10} {dim_str:>16}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()
    all_outputs: list[Path] = []

    for slide_cfg in SLIDES:
        all_outputs.extend(process_slide(slide_cfg))

    report(all_outputs)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
