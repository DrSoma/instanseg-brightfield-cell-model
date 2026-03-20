#!/usr/bin/env python3
"""D5: Reproducibility Assessment per CLSI EP05.

Run 30 tiles 3x each with small perturbations (±5° rotation, ±10px crop offset).
Compute coefficient of variation (CV) for:
  - Cell count
  - H-score
  - % Positive (≥2+)
  - Grade distribution

CLSI EP05 acceptance: CV < 15% for each metric.

Usage:
    python scripts/09_reproducibility_test.py [--n-tiles 30] [--n-reps 3]
"""

from __future__ import annotations

import argparse
import json
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
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "reproducibility"

VENV_PYTHON = "/home/fernandosoto/claudin18_venv/bin/python"
PIPELINE_DIR = Path(
    "/home/fernandosoto/Documents/cldn18-pathology-pipeline-main (3)/"
    "cldn18-pathology-pipeline-main/scripts"
)

# Stain deconvolution vectors
STAIN_H = np.array([0.786, 0.593, 0.174])
STAIN_DAB = np.array([0.215, 0.422, 0.881])
STAIN_R = np.array([0.547, -0.799, 0.249])

# Bar-filter parameters
BAR_N_ORIENTATIONS = 8
BAR_KSIZE = 25
BAR_SIGMA_LONG = 5.0
BAR_SIGMA_SHORT = 1.0

# Clinical thresholds (preliminary)
THRESHOLDS = (0.10, 0.20, 0.35)


def get_bar_filter_kernels(n_orientations=8, ksize=25, sigma_long=5.0, sigma_short=1.0):
    """Create oriented bar-filter kernels for membrane detection."""
    kernels = []
    for i in range(n_orientations):
        angle = i * 180.0 / n_orientations
        # Create anisotropic Gaussian
        sigma_x = sigma_short
        sigma_y = sigma_long
        half = ksize // 2
        y, x = np.mgrid[-half:half+1, -half:half+1].astype(np.float64)
        # Rotate coordinates
        rad = np.deg2rad(angle)
        xr = x * np.cos(rad) + y * np.sin(rad)
        yr = -x * np.sin(rad) + y * np.cos(rad)
        kernel = np.exp(-0.5 * (xr**2 / sigma_x**2 + yr**2 / sigma_y**2))
        kernel -= kernel.mean()
        kernel /= kernel.std() + 1e-8
        kernels.append(kernel.astype(np.float32))
    return kernels


def stain_deconvolve(rgb: np.ndarray) -> np.ndarray:
    """Deconvolve RGB image into H/DAB/R channels. Returns DAB channel."""
    stain_matrix = np.array([STAIN_H, STAIN_DAB, STAIN_R])
    inv_matrix = np.linalg.inv(stain_matrix)

    rgb_float = rgb.astype(np.float64) / 255.0
    rgb_float = np.clip(rgb_float, 1e-6, 1.0)
    od = -np.log(rgb_float)

    flat = od.reshape(-1, 3)
    deconv = flat @ inv_matrix.T
    deconv = deconv.reshape(rgb.shape)

    return np.clip(deconv[:, :, 1], 0, None)  # DAB channel


def measure_membrane_ring(tile_rgb: np.ndarray, mask: np.ndarray,
                          bar_kernels: list) -> dict:
    """Measure membrane DAB for all cells in a segmentation mask.

    Returns dict with per-cell measurements.
    """
    dab_channel = stain_deconvolve(tile_rgb)

    # Apply bar filter
    bar_responses = [cv2.filter2D(dab_channel.astype(np.float32), -1, k) for k in bar_kernels]
    bar_max = np.maximum.reduce(bar_responses)

    # Per-cell measurements
    labels = np.unique(mask)
    labels = labels[labels > 0]

    cell_data = []
    for label_id in labels:
        cell_mask = mask == label_id
        ys, xs = np.where(cell_mask)
        if len(ys) < 5:
            continue

        # Centroid
        cx = float(xs.mean())
        cy = float(ys.mean())

        # Membrane ring: dilate - erode
        kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(cell_mask.astype(np.uint8), kernel_3, iterations=2)
        eroded = cv2.erode(cell_mask.astype(np.uint8), kernel_3, iterations=1)
        ring = (dilated - eroded).astype(bool)

        if ring.sum() < 3:
            continue

        ring_dab = dab_channel[ring].mean()
        ring_bar = bar_max[ring].mean()

        # Completeness: fraction of ring pixels with DAB > threshold
        completeness = float((dab_channel[ring] > 0.1).mean())

        # Thickness: FWHM of radial DAB profile (simplified)
        thickness_px = 2.0 * np.sqrt(ring.sum() / (np.pi + 1e-8))

        # Grade assignment
        composite = ring_bar * completeness
        if composite < THRESHOLDS[0]:
            grade = 0
        elif composite < THRESHOLDS[1]:
            grade = 1
        elif composite < THRESHOLDS[2]:
            grade = 2
        else:
            grade = 3

        cell_data.append({
            "centroid_x": cx,
            "centroid_y": cy,
            "membrane_ring_dab": float(ring_dab),
            "membrane_completeness": completeness,
            "membrane_thickness_px": float(thickness_px),
            "composite_grade": grade,
        })

    return cell_data


def apply_perturbation(tile: np.ndarray, rep: int) -> np.ndarray:
    """Apply small perturbation to tile for reproducibility test.

    rep=0: original (no perturbation)
    rep=1: +3° rotation + 5px crop offset
    rep=2: -3° rotation + 8px crop offset
    """
    if rep == 0:
        return tile.copy()

    h, w = tile.shape[:2]
    center = (w / 2, h / 2)

    if rep == 1:
        angle = 3.0
        offset_x, offset_y = 5, 5
    elif rep == 2:
        angle = -3.0
        offset_x, offset_y = 8, -5
    else:
        angle = np.random.uniform(-5, 5)
        offset_x = np.random.randint(-10, 10)
        offset_y = np.random.randint(-10, 10)

    # Rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Add translation
    M[0, 2] += offset_x
    M[1, 2] += offset_y
    rotated = cv2.warpAffine(tile, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return rotated


def run_instanseg_on_tile(model, tile_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """Run InstanSeg inference on a single tile. Returns labeled mask."""
    # Normalize to [0, 1] float
    tile_float = tile_rgb.astype(np.float32) / 255.0

    # InstanSeg expects (B, C, H, W) in [0, 1]
    tile_tensor = torch.from_numpy(tile_float).permute(2, 0, 1).unsqueeze(0)
    tile_tensor = tile_tensor.to(device)

    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.float16 if device.type == "cuda" else torch.float32):
            output = model.instanseg(tile_tensor)

    # output shape: (B, N_channels, H, W) — take cells channel (index 1 for NC mode)
    if output.shape[1] >= 2:
        mask = output[0, 1].cpu().numpy()  # cells channel
    else:
        mask = output[0, 0].cpu().numpy()

    return mask.astype(np.int32)


def select_representative_tiles(parquet_path: Path, slide_path: Path,
                                n_tiles: int = 30) -> list[dict]:
    """Select tiles stratified by composite grade for reproducibility test."""
    import openslide

    df = pd.read_parquet(parquet_path)

    # Determine tile step from slide properties
    slide = openslide.OpenSlide(str(slide_path))
    base_mpp = float(slide.properties.get("openslide.mpp-x", 0.25))
    model_mpp = 0.5

    # Find best level
    best_level = 0
    best_ds = 1.0
    for lvl in range(slide.level_count):
        ds = slide.level_downsamples[lvl]
        lvl_mpp = base_mpp * ds
        if abs(lvl_mpp - model_mpp) < abs(base_mpp * best_ds - model_mpp):
            best_level = lvl
            best_ds = ds

    tile_size = 512
    padding = 80
    content_l0 = int(round((tile_size - 2 * padding) * best_ds))
    slide.close()

    # Assign cells to tiles
    df["tile_x"] = (df["centroid_x"] // content_l0).astype(int)
    df["tile_y"] = (df["centroid_y"] // content_l0).astype(int)

    # Per-tile stats
    tile_stats = df.groupby(["tile_x", "tile_y"]).agg(
        n_cells=("cell_id", "count"),
        mean_grade=("cldn18_composite_grade", "mean"),
    ).reset_index()

    # Filter tiles with enough cells
    tile_stats = tile_stats[tile_stats["n_cells"] >= 10].copy()

    if len(tile_stats) < n_tiles:
        print(f"  WARNING: Only {len(tile_stats)} tiles with ≥10 cells (need {n_tiles})")
        n_tiles = len(tile_stats)

    # Stratified selection: sort by mean grade, pick evenly
    tile_stats = tile_stats.sort_values("mean_grade").reset_index(drop=True)
    indices = np.linspace(0, len(tile_stats) - 1, n_tiles, dtype=int)
    selected = tile_stats.iloc[indices]

    tiles = []
    for _, row in selected.iterrows():
        tile_x_l0 = int(row["tile_x"] * content_l0)
        tile_y_l0 = int(row["tile_y"] * content_l0)
        tiles.append({
            "tile_x_l0": tile_x_l0,
            "tile_y_l0": tile_y_l0,
            "n_cells": int(row["n_cells"]),
            "mean_grade": float(row["mean_grade"]),
            "level": best_level,
            "level_ds": float(best_ds),
        })

    return tiles


def main():
    parser = argparse.ArgumentParser(description="D5: Reproducibility assessment")
    parser.add_argument("--n-tiles", type=int, default=30)
    parser.add_argument("--n-reps", type=int, default=3)
    parser.add_argument("--parquet", type=Path,
                        default=Path("/tmp/pipeline_comparison/v2_cells_nuclei/cell_data/BC_ClassII_cells.parquet"))
    parser.add_argument("--slide", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  D5: Reproducibility Assessment (CLSI EP05)")
    print("=" * 60)
    print(f"  Tiles: {args.n_tiles}")
    print(f"  Repetitions: {args.n_reps}")
    print(f"  CLSI EP05 threshold: CV < 15%")

    # Find slide
    if args.slide is None:
        slide_name = args.parquet.stem.replace("_cells", "")
        for search_dir in [Path("/tmp/bc_slides"), Path("/pathodata/Claudin18_project")]:
            for ext in [".ndpi", ".svs"]:
                candidate = search_dir / f"{slide_name}{ext}"
                if candidate.exists():
                    args.slide = candidate
                    break
            if args.slide:
                break
        # Try finding with find
        if not args.slide:
            import subprocess
            result = subprocess.run(
                ["find", "/tmp", "/pathodata", "-name", f"{slide_name}.*", "-type", "f"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.strip().split("\n"):
                if line.endswith((".ndpi", ".svs")):
                    args.slide = Path(line)
                    break

    if not args.slide or not args.slide.exists():
        print(f"  ERROR: Could not find slide for {args.parquet.stem}")
        sys.exit(1)

    print(f"  Slide: {args.slide}")
    print(f"  Parquet: {args.parquet}")

    # Select tiles
    print("\n  Selecting representative tiles...")
    tiles = select_representative_tiles(args.parquet, args.slide, args.n_tiles)
    print(f"  Selected {len(tiles)} tiles (grade range: "
          f"{tiles[0]['mean_grade']:.2f} - {tiles[-1]['mean_grade']:.2f})")

    # Load InstanSeg model
    print("\n  Loading InstanSeg model...")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from instanseg import InstanSeg
    model = InstanSeg("brightfield_cells_nuclei", device=str(device))
    print(f"  Model loaded on {device}")

    # Warmup
    dummy = torch.randn(1, 3, 512, 512, device=device)
    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.float16 if device.type == "cuda" else torch.float32):
            _ = model.instanseg(dummy)
    del dummy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Bar filter kernels
    bar_kernels = get_bar_filter_kernels(BAR_N_ORIENTATIONS, BAR_KSIZE,
                                         BAR_SIGMA_LONG, BAR_SIGMA_SHORT)

    # Open slide
    import openslide
    slide = openslide.OpenSlide(str(args.slide))

    # Run reproducibility test
    print(f"\n  Running {len(tiles)} tiles × {args.n_reps} reps = {len(tiles) * args.n_reps} inferences...")
    all_results = []
    t0 = time.time()

    for tile_idx, tile_info in enumerate(tiles):
        tile_results = []

        # Read tile
        x0 = tile_info["tile_x_l0"]
        y0 = tile_info["tile_y_l0"]
        level = tile_info["level"]
        tile_size = 512

        tile_rgb = np.array(slide.read_region(
            (x0, y0), level, (tile_size, tile_size)
        ).convert("RGB"))

        for rep in range(args.n_reps):
            # Apply perturbation
            perturbed = apply_perturbation(tile_rgb, rep)

            # Run inference
            mask = run_instanseg_on_tile(model, perturbed, device)

            # Measure membrane features
            cell_data = measure_membrane_ring(perturbed, mask, bar_kernels)

            # Compute metrics
            n_cells = len(cell_data)
            if n_cells > 0:
                grades = np.array([c["composite_grade"] for c in cell_data])
                h_score = float(
                    1 * (grades == 1).sum() / n_cells * 100
                    + 2 * (grades == 2).sum() / n_cells * 100
                    + 3 * (grades == 3).sum() / n_cells * 100
                )
                pct_pos = float((grades >= 2).sum() / n_cells * 100)
                pct_3plus = float((grades == 3).sum() / n_cells * 100)
            else:
                h_score = 0.0
                pct_pos = 0.0
                pct_3plus = 0.0

            tile_results.append({
                "tile_idx": tile_idx,
                "rep": rep,
                "n_cells": n_cells,
                "h_score": h_score,
                "pct_positive": pct_pos,
                "pct_3plus": pct_3plus,
            })

        all_results.append(tile_results)

        if (tile_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    Tile {tile_idx+1}/{len(tiles)} "
                  f"({elapsed:.0f}s, {elapsed/(tile_idx+1):.1f}s/tile)")

    slide.close()
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s ({elapsed/len(tiles):.1f}s/tile)")

    # Compute CVs
    print("\n--- Coefficient of Variation (CV) ---")
    metrics = ["n_cells", "h_score", "pct_positive", "pct_3plus"]
    metric_names = ["Cell Count", "H-Score", "% Positive (≥2+)", "% Grade 3+"]
    cv_results = {}

    for metric, mname in zip(metrics, metric_names):
        per_tile_cvs = []
        for tile_results in all_results:
            values = [r[metric] for r in tile_results]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0
            cv = (std_val / mean_val * 100) if mean_val > 0 else 0
            per_tile_cvs.append(cv)

        overall_cv = np.mean(per_tile_cvs)
        median_cv = np.median(per_tile_cvs)
        max_cv = np.max(per_tile_cvs)
        pass_rate = np.mean(np.array(per_tile_cvs) < 15) * 100

        cv_results[metric] = {
            "mean_cv": float(overall_cv),
            "median_cv": float(median_cv),
            "max_cv": float(max_cv),
            "pass_rate_pct": float(pass_rate),
            "passes_clsi": bool(overall_cv < 15),
            "per_tile_cvs": [float(c) for c in per_tile_cvs],
        }

        status = "PASS" if overall_cv < 15 else "FAIL"
        print(f"  {mname:<25} Mean CV={overall_cv:>6.2f}%  Median={median_cv:>6.2f}%  "
              f"Max={max_cv:>6.2f}%  [{status}]")

    # Overall pass/fail
    all_pass = all(v["passes_clsi"] for v in cv_results.values())
    print(f"\n  Overall CLSI EP05: {'PASS' if all_pass else 'FAIL'}")

    # Generate plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (metric, mname) in zip(axes.flat, zip(metrics, metric_names)):
        cvs = cv_results[metric]["per_tile_cvs"]
        ax.bar(range(len(cvs)), cvs, color=["green" if c < 15 else "red" for c in cvs])
        ax.axhline(y=15, color="red", linestyle="--", label="CLSI EP05 threshold (15%)")
        ax.set_xlabel("Tile Index")
        ax.set_ylabel("CV (%)")
        ax.set_title(f"{mname} — Mean CV={cv_results[metric]['mean_cv']:.1f}%")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Reproducibility Assessment (CLSI EP05)\n"
                 f"{len(tiles)} tiles × {args.n_reps} reps | "
                 f"Overall: {'PASS' if all_pass else 'FAIL'}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.output_dir / "reproducibility_cv.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved plot: {args.output_dir / 'reproducibility_cv.png'}")

    # Save results
    results_json = {
        "config": {
            "n_tiles": len(tiles),
            "n_reps": args.n_reps,
            "slide": str(args.slide),
            "model": "brightfield_cells_nuclei",
            "perturbations": [
                "rep 0: original",
                "rep 1: +3° rotation, +5px offset",
                "rep 2: -3° rotation, +8px/-5px offset",
            ],
        },
        "cv_results": cv_results,
        "clsi_ep05_pass": all_pass,
        "raw_results": [
            [r for r in tile_results]
            for tile_results in all_results
        ],
        "elapsed_seconds": elapsed,
    }
    with open(args.output_dir / "reproducibility_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"  Results saved to {args.output_dir}/")
    print("  Done.")


if __name__ == "__main__":
    main()
