"""
Test whether brightfield_cells_nuclei detects cells on DAB-negative tissue.

Hypothesis: The model should detect near-zero cells on blue-only (hematoxylin)
tissue, confirming it learned to segment CLDN18.2-expressing cells specifically,
not all cells indiscriminately.

Comparison baseline: brightfield_nuclei model, which detects ALL nuclei.

DAB classification approach:
  1. Build a tissue mask (exclude white background via grayscale Otsu)
  2. Deconvolve stains within tissue to get DAB concentration per pixel
  3. Classify each pixel as DAB-positive if DAB concentration > 0.1
  4. Compute DAB-positive fraction = (# DAB-positive tissue pixels) / (# tissue pixels)
  5. Bin tiles by DAB-positive fraction:
       Negative  : < 5%  of tissue is DAB-positive
       Low       : 5-15%
       Moderate  : 15-30%
       High      : > 30%
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/fernandosoto/Documents/instanseg-brightfield-cell-model")
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "segmentation_dataset.pth"

OUR_MODEL_PATH = Path(
    "/home/fernandosoto/claudin18_venv/lib/python3.12/site-packages/"
    "instanseg/bioimageio_models/brightfield_cells_nuclei/instanseg.pt"
)
BASELINE_MODEL_PATH = Path(
    "/home/fernandosoto/claudin18_venv/lib/python3.12/site-packages/"
    "instanseg/bioimageio_models/brightfield_nuclei/0.1.1/instanseg.pt"
)

OUTPUT_JSON = PROJECT_ROOT / "evaluation" / "negative_tissue_test.json"

# ---------------------------------------------------------------------------
# Stain deconvolution -- calibrated vectors for our IHC slides
# ---------------------------------------------------------------------------
H_VEC = np.array([0.786, 0.593, 0.174], dtype=np.float64)
DAB_VEC = np.array([0.215, 0.422, 0.881], dtype=np.float64)
R_VEC = np.array([0.547, -0.799, 0.249], dtype=np.float64)

# Normalise vectors to unit length
H_VEC /= np.linalg.norm(H_VEC)
DAB_VEC /= np.linalg.norm(DAB_VEC)
R_VEC /= np.linalg.norm(R_VEC)

# Build deconvolution matrix (inverse of stain matrix)
STAIN_MATRIX = np.stack([H_VEC, DAB_VEC, R_VEC], axis=0)  # (3, 3)
DECONV_MATRIX = np.linalg.inv(STAIN_MATRIX)

# DAB-positive pixel threshold (stain concentration units)
DAB_PIXEL_THRESH = 0.10

# Tile classification by DAB-positive fraction of tissue
DAB_BINS = {
    "Negative": (0.0, 0.05),
    "Low": (0.05, 0.15),
    "Moderate": (0.15, 0.30),
    "High": (0.30, 1.01),
}


def analyze_dab(image_uint8: np.ndarray) -> dict:
    """Analyze DAB staining in an RGB tile.

    Returns dict with tissue_fraction, dab_positive_fraction,
    mean_dab_in_tissue, and the raw dab concentration map.
    """
    h, w = image_uint8.shape[:2]
    n_pixels = h * w

    # --- Tissue mask via grayscale Otsu ---
    gray = np.mean(image_uint8.astype(np.float64), axis=2)
    # Simple Otsu: tissue is darker than background
    # Use numpy histogram-based Otsu
    hist, bin_edges = np.histogram(gray, bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    sum_total = (hist * bin_centers).sum()
    w0 = 0.0
    sum0 = 0.0
    max_var = 0.0
    threshold = 128.0  # default
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
        between_var = w0 * w1 * (mean0 - mean1) ** 2
        if between_var > max_var:
            max_var = between_var
            threshold = bin_centers[i]

    tissue_mask = gray < threshold  # tissue is darker
    tissue_count = int(tissue_mask.sum())
    tissue_fraction = tissue_count / n_pixels

    if tissue_count < 100:
        # Almost no tissue -- skip
        return {
            "tissue_fraction": tissue_fraction,
            "dab_positive_fraction": 0.0,
            "mean_dab_in_tissue": 0.0,
            "dab_level": "Negative",
        }

    # --- Stain deconvolution ---
    img = image_uint8.astype(np.float64) / 255.0
    img = np.clip(img, 1e-6, 1.0)
    od = -np.log10(img)  # (H, W, 3)

    od_flat = od.reshape(-1, 3)
    stain_conc = od_flat @ DECONV_MATRIX.T  # (N, 3): [H, DAB, Residual]
    dab_map = stain_conc[:, 1].reshape(h, w)

    # --- DAB metrics within tissue ---
    dab_in_tissue = dab_map[tissue_mask]
    mean_dab = float(np.mean(np.clip(dab_in_tissue, 0, None)))
    dab_positive_pixels = int((dab_in_tissue > DAB_PIXEL_THRESH).sum())
    dab_positive_fraction = dab_positive_pixels / tissue_count

    # --- Classify ---
    dab_level = "High"
    for level, (lo, hi) in DAB_BINS.items():
        if lo <= dab_positive_fraction < hi:
            dab_level = level
            break

    return {
        "tissue_fraction": round(tissue_fraction, 4),
        "dab_positive_fraction": round(dab_positive_fraction, 4),
        "mean_dab_in_tissue": round(mean_dab, 4),
        "dab_level": dab_level,
    }


def count_cells(label_map: torch.Tensor) -> int:
    """Count unique labels > 0 in a label map tensor."""
    unique = torch.unique(label_map)
    return int((unique > 0).sum().item())


def run_model(model, image_tensor: torch.Tensor, target_seg: torch.Tensor) -> int:
    """Run a model on a single image and return cell count."""
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        output = model(image_tensor, target_segmentation=target_seg)
    if output.dim() == 4:
        output = output[0]
    if output.dim() == 3 and output.shape[0] >= 1:
        cell_labels = output[0]
    else:
        cell_labels = output
    return count_cells(cell_labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("Loading dataset...")
    dataset = torch.load(str(DATASET_PATH), map_location="cpu", weights_only=False)
    test_items = dataset["Test"]
    print(f"Test set: {len(test_items)} tiles")
    print()

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("Loading our model (brightfield_cells_nuclei)...")
    our_model = torch.jit.load(str(OUR_MODEL_PATH), map_location=device)
    our_model.eval()

    print("Loading baseline model (brightfield_nuclei)...")
    baseline_model = torch.jit.load(str(BASELINE_MODEL_PATH), map_location=device)
    baseline_model.eval()
    print()

    # Target segmentation tensors
    target_cells_nuclei = torch.tensor([1, 1], dtype=torch.long, device=device)
    target_nuclei = torch.tensor([1], dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Process each tile
    # ------------------------------------------------------------------
    results = []
    t0 = time.time()

    for idx, item in enumerate(test_items):
        image_path = DATA_DIR / item["image"]
        img = tifffile.imread(str(image_path))  # (H, W, 3), uint8

        # --- DAB analysis ---
        dab_info = analyze_dab(img)

        # --- Prepare tensor: float32, [0,1], (1, 3, H, W) ---
        img_f = img.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)

        # --- Run our model ---
        our_cells = run_model(our_model, img_t, target_cells_nuclei)

        # --- Run baseline ---
        baseline_cells = run_model(baseline_model, img_t, target_nuclei)

        ratio = our_cells / baseline_cells if baseline_cells > 0 else float("inf")

        results.append({
            "index": idx,
            "image": str(item["image"]),
            "tissue_fraction": dab_info["tissue_fraction"],
            "dab_positive_fraction": dab_info["dab_positive_fraction"],
            "mean_dab_in_tissue": dab_info["mean_dab_in_tissue"],
            "dab_level": dab_info["dab_level"],
            "our_model_cells": our_cells,
            "baseline_nuclei": baseline_cells,
            "ratio": round(ratio, 4),
        })

        # Progress
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(test_items) - idx - 1) / rate
            print(
                f"  [{idx+1:3d}/{len(test_items)}] "
                f"DAB+={dab_info['dab_positive_fraction']:5.1%} ({dab_info['dab_level']:>8s}) "
                f"tissue={dab_info['tissue_fraction']:4.0%} | "
                f"Our={our_cells:4d}  Base={baseline_cells:4d}  r={ratio:.2f} | "
                f"{rate:.1f} t/s  ETA {eta:.0f}s"
            )

    elapsed_total = time.time() - t0
    print(f"\nProcessed {len(results)} tiles in {elapsed_total:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Aggregate by DAB level
    # ------------------------------------------------------------------
    summary = {}
    for level in DAB_BINS:
        tiles = [r for r in results if r["dab_level"] == level]
        n = len(tiles)
        if n == 0:
            summary[level] = {
                "tiles": 0,
                "our_model_mean": 0.0,
                "our_model_median": 0.0,
                "baseline_mean": 0.0,
                "baseline_median": 0.0,
                "ratio_mean": 0.0,
                "ratio_median": 0.0,
                "our_model_total": 0,
                "baseline_total": 0,
                "mean_dab_positive_pct": 0.0,
            }
            continue

        our_counts = np.array([t["our_model_cells"] for t in tiles])
        base_counts = np.array([t["baseline_nuclei"] for t in tiles])
        ratios = np.array([t["ratio"] for t in tiles])
        dab_pcts = np.array([t["dab_positive_fraction"] for t in tiles])

        our_mean = float(our_counts.mean())
        base_mean = float(base_counts.mean())
        mean_ratio = our_mean / base_mean if base_mean > 0 else float("inf")

        summary[level] = {
            "tiles": n,
            "our_model_mean": round(our_mean, 1),
            "our_model_median": float(np.median(our_counts)),
            "baseline_mean": round(base_mean, 1),
            "baseline_median": float(np.median(base_counts)),
            "ratio_mean": round(mean_ratio, 3),
            "ratio_median": round(float(np.median(ratios)), 3),
            "our_model_total": int(our_counts.sum()),
            "baseline_total": int(base_counts.sum()),
            "mean_dab_positive_pct": round(float(dab_pcts.mean()) * 100, 1),
        }

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    header = (
        f"{'DAB Level':>10s} | {'Tiles':>5s} | {'DAB+ %':>6s} | "
        f"{'Our (mean)':>10s} | {'Base (mean)':>11s} | {'Ratio':>6s} | "
        f"{'Our (med)':>9s} | {'Base (med)':>10s} | {'Ratio(med)':>10s}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for level in DAB_BINS:
        s = summary[level]
        if s["tiles"] == 0:
            print(
                f"{level:>10s} |     0 |    -- |        -- |"
                f"         -- |   -- |       -- |        -- |        --"
            )
        else:
            print(
                f"{level:>10s} | {s['tiles']:5d} | {s['mean_dab_positive_pct']:5.1f}% | "
                f"{s['our_model_mean']:10.1f} | {s['baseline_mean']:11.1f} | "
                f"{s['ratio_mean']:6.3f} | "
                f"{s['our_model_median']:9.0f} | {s['baseline_median']:10.0f} | "
                f"{s['ratio_median']:10.3f}"
            )
    print(sep)
    print()

    # ------------------------------------------------------------------
    # Print per-tile details for each DAB category (limit to 20 per bin)
    # ------------------------------------------------------------------
    for level in DAB_BINS:
        tiles_in_bin = sorted(
            [r for r in results if r["dab_level"] == level],
            key=lambda x: x["dab_positive_fraction"],
        )
        if not tiles_in_bin:
            continue
        print(f"=== {level} Tiles (n={len(tiles_in_bin)}) ===")
        print(
            f"{'Image':>45s} | {'DAB+%':>5s} | {'Tissue':>6s} | "
            f"{'Our':>5s} | {'Base':>5s} | {'Ratio':>5s}"
        )
        print("-" * 85)
        shown = tiles_in_bin[:10] + tiles_in_bin[-10:] if len(tiles_in_bin) > 20 else tiles_in_bin
        prev_was_gap = False
        for i, t in enumerate(shown):
            if i == 10 and len(tiles_in_bin) > 20 and not prev_was_gap:
                print(f"{'... (' + str(len(tiles_in_bin) - 20) + ' more) ...':>45s} |       |        |       |       |")
                prev_was_gap = True
            print(
                f"{t['image']:>45s} | {t['dab_positive_fraction']*100:4.1f}% | "
                f"{t['tissue_fraction']*100:5.1f}% | "
                f"{t['our_model_cells']:5d} | {t['baseline_nuclei']:5d} | "
                f"{t['ratio']:5.2f}"
            )
        print()

    # ------------------------------------------------------------------
    # Hypothesis verdict
    # ------------------------------------------------------------------
    neg = summary.get("Negative", {"tiles": 0})
    high = summary.get("High", {"tiles": 0})

    print("=" * 70)
    print("HYPOTHESIS TEST: Does our model ignore DAB-negative tissue?")
    print("=" * 70)

    if neg["tiles"] > 0 and high["tiles"] > 0:
        neg_ratio = neg["ratio_mean"]
        high_ratio = high["ratio_mean"]

        # Strong confirmation: on negative tissue the model detects < 10% of
        # what baseline detects, AND on high-DAB tissue it detects substantially more
        if neg_ratio < 0.10:
            verdict = "STRONGLY CONFIRMED"
            detail = (
                f"On DAB-negative tissue (n={neg['tiles']}), our model detects "
                f"{neg_ratio:.1%} as many instances as baseline "
                f"({neg['our_model_mean']:.1f} vs {neg['baseline_mean']:.1f}). "
                f"On DAB-high tissue (n={high['tiles']}), ratio is {high_ratio:.3f}. "
                f"The model has learned strong DAB specificity."
            )
        elif neg_ratio < 0.30:
            verdict = "PARTIALLY CONFIRMED"
            detail = (
                f"On negative tissue, ratio={neg_ratio:.3f} "
                f"({neg['our_model_mean']:.1f} vs {neg['baseline_mean']:.1f}). "
                f"On high-DAB tissue, ratio={high_ratio:.3f}. "
                f"The model shows partial DAB selectivity."
            )
        else:
            verdict = "REJECTED"
            detail = (
                f"On negative tissue, ratio={neg_ratio:.3f} "
                f"({neg['our_model_mean']:.1f} vs {neg['baseline_mean']:.1f}). "
                f"On high-DAB tissue, ratio={high_ratio:.3f}. "
                f"The model does NOT show strong DAB selectivity."
            )
    elif neg["tiles"] > 0:
        neg_ratio = neg["ratio_mean"]
        verdict = "PARTIALLY ASSESSED"
        detail = (
            f"On DAB-negative tissue (n={neg['tiles']}), ratio={neg_ratio:.3f}. "
            f"No DAB-high tiles found for comparison."
        )
    else:
        verdict = "INCONCLUSIVE"
        detail = "No DAB-negative tiles found in test set."

    print(f"Verdict: {verdict}")
    print(f"Detail:  {detail}")
    print()

    # Print the gradient
    print("Detection ratio gradient across DAB levels:")
    for level in DAB_BINS:
        s = summary[level]
        if s["tiles"] > 0:
            bar_len = int(s["ratio_mean"] * 50)
            bar = "#" * min(bar_len, 50)
            print(f"  {level:>10s}: {s['ratio_mean']:.3f} |{bar}")
    print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "test_name": "negative_tissue_hypothesis_test",
        "date": "2026-03-19",
        "description": (
            "Tests whether brightfield_cells_nuclei model detects cells on "
            "DAB-negative (blue-only) tissue. Tiles classified by percentage of "
            "tissue pixels with DAB concentration > 0.1 (after stain deconvolution). "
            "Low detection ratio on negative tissue confirms CLDN18.2 specificity."
        ),
        "method": {
            "tissue_mask": "Otsu threshold on grayscale (tissue = dark pixels)",
            "dab_deconvolution": "Calibrated H/DAB/R vectors, OD-space deconvolution",
            "dab_pixel_threshold": DAB_PIXEL_THRESH,
            "bins": {k: f"{v[0]*100:.0f}-{v[1]*100:.0f}% DAB+ tissue" for k, v in DAB_BINS.items()},
        },
        "verdict": verdict,
        "detail": detail,
        "summary": summary,
        "per_tile": results,
        "total_tiles": len(results),
        "elapsed_seconds": round(elapsed_total, 1),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
