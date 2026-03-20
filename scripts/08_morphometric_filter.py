#!/usr/bin/env python3
"""D3: Per-cell morphometric filtering for CLDN18.2 pipeline.

Filters non-epithelial cells using geometric criteria:
- Small cells (lymphocytes): area < threshold
- Irregular shapes: solidity < threshold
- Elongated cells (fibroblasts): aspect ratio > threshold

Reports impact on grade distribution and H-scores at multiple thresholds.
Saves filtered results to evaluation/morphometric_filtering/.

Usage:
    python scripts/08_morphometric_filter.py [--input-dir /tmp/pipeline_comparison/v2_cells_nuclei/cell_data]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "morphometric_filtering"


def compute_hscore(grades: np.ndarray) -> float:
    if len(grades) == 0:
        return 0.0
    n = len(grades)
    return float(
        1 * (grades == 1).sum() / n * 100
        + 2 * (grades == 2).sum() / n * 100
        + 3 * (grades == 3).sum() / n * 100
    )


def compute_pct_positive(grades: np.ndarray) -> float:
    if len(grades) == 0:
        return 0.0
    return float((grades >= 2).sum() / len(grades) * 100)


def grade_distribution(grades: np.ndarray) -> dict:
    total = len(grades)
    if total == 0:
        return {0: 0, 1: 0, 2: 0, 3: 0}
    return {
        g: int((grades == g).sum())
        for g in range(4)
    }


def apply_filter(df: pd.DataFrame, min_area: float, min_solidity: float,
                 max_aspect: float) -> pd.DataFrame:
    """Apply morphometric filters. Returns filtered DataFrame."""
    mask = pd.Series(True, index=df.index)

    if min_area > 0:
        mask &= df["area_um2"] >= min_area

    if min_solidity > 0:
        # Clamp solidity to [0, 1] range (values >1 are numerical artifacts)
        sol = df["solidity"].clip(upper=1.0)
        mask &= sol >= min_solidity

    if max_aspect < np.inf:
        aspect = df["max_diameter_um"] / df["min_diameter_um"].clip(lower=0.01)
        mask &= aspect <= max_aspect

    return df[mask]


def analyze_threshold_sensitivity(df: pd.DataFrame, slide_name: str) -> list[dict]:
    """Test multiple threshold combinations and report impact."""
    configs = [
        {"name": "No filter", "min_area": 0, "min_solidity": 0, "max_aspect": np.inf},
        {"name": "Mild (area≥15)", "min_area": 15, "min_solidity": 0, "max_aspect": np.inf},
        {"name": "Moderate (area≥25, sol≥0.85)", "min_area": 25, "min_solidity": 0.85, "max_aspect": 3.0},
        {"name": "Standard (area≥25, sol≥0.9, asp≤2.5)", "min_area": 25, "min_solidity": 0.9, "max_aspect": 2.5},
        {"name": "Strict (area≥35, sol≥0.92, asp≤2.0)", "min_area": 35, "min_solidity": 0.92, "max_aspect": 2.0},
        {"name": "Debate (area≥80, sol≥0.9, asp≤2.0)", "min_area": 80, "min_solidity": 0.9, "max_aspect": 2.0},
    ]

    results = []
    grades_col = "cldn18_composite_grade"

    for cfg in configs:
        filtered = apply_filter(df, cfg["min_area"], cfg["min_solidity"], cfg["max_aspect"])
        grades = filtered[grades_col].values.astype(int)

        result = {
            "slide": slide_name,
            "filter": cfg["name"],
            "min_area": cfg["min_area"],
            "min_solidity": cfg["min_solidity"],
            "max_aspect": float(cfg["max_aspect"]),
            "total_before": len(df),
            "total_after": len(filtered),
            "pct_retained": len(filtered) / len(df) * 100 if len(df) > 0 else 0,
            "h_score": compute_hscore(grades),
            "pct_positive": compute_pct_positive(grades),
            "grade_dist": grade_distribution(grades),
        }
        results.append(result)

    return results


def plot_sensitivity(all_results: list[dict], output_path: Path):
    """Plot threshold sensitivity analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    slides = sorted(set(r["slide"] for r in all_results))
    colors = plt.cm.tab10(np.linspace(0, 1, len(slides)))

    for slide_idx, slide in enumerate(slides):
        slide_results = [r for r in all_results if r["slide"] == slide]
        labels = [r["filter"] for r in slide_results]
        x = range(len(labels))

        # Cells retained
        axes[0, 0].plot(x, [r["pct_retained"] for r in slide_results],
                        "o-", label=slide, color=colors[slide_idx])
        axes[0, 0].set_ylabel("% Cells Retained")
        axes[0, 0].set_title("Cells Retained by Filter")

        # H-score
        axes[0, 1].plot(x, [r["h_score"] for r in slide_results],
                        "o-", label=slide, color=colors[slide_idx])
        axes[0, 1].set_ylabel("H-Score")
        axes[0, 1].set_title("H-Score by Filter")

        # % Positive
        axes[1, 0].plot(x, [r["pct_positive"] for r in slide_results],
                        "o-", label=slide, color=colors[slide_idx])
        axes[1, 0].set_ylabel("% Positive (≥2+)")
        axes[1, 0].set_title("Positivity by Filter")

        # 3+ percentage
        pct_3plus = []
        for r in slide_results:
            total = r["total_after"]
            n3 = r["grade_dist"].get(3, 0)
            pct_3plus.append(n3 / total * 100 if total > 0 else 0)
        axes[1, 1].plot(x, pct_3plus, "o-", label=slide, color=colors[slide_idx])
        axes[1, 1].set_ylabel("% Grade 3+")
        axes[1, 1].set_title("3+ Population by Filter")

    for ax in axes.flat:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Morphometric Filter Sensitivity Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def morphometric_summary_stats(df: pd.DataFrame, slide_name: str) -> dict:
    """Compute morphometric feature distributions."""
    aspect = df["max_diameter_um"] / df["min_diameter_um"].clip(lower=0.01)
    sol_clamped = df["solidity"].clip(upper=1.0)

    return {
        "slide": slide_name,
        "n_cells": len(df),
        "area_um2": {
            "mean": float(df["area_um2"].mean()),
            "median": float(df["area_um2"].median()),
            "p5": float(df["area_um2"].quantile(0.05)),
            "p25": float(df["area_um2"].quantile(0.25)),
            "p75": float(df["area_um2"].quantile(0.75)),
            "p95": float(df["area_um2"].quantile(0.95)),
        },
        "solidity": {
            "mean": float(sol_clamped.mean()),
            "median": float(sol_clamped.median()),
            "pct_below_0.9": float((sol_clamped < 0.9).mean() * 100),
            "pct_above_1.0": float((df["solidity"] > 1.0).mean() * 100),
        },
        "aspect_ratio": {
            "mean": float(aspect.mean()),
            "median": float(aspect.median()),
            "p95": float(aspect.quantile(0.95)),
            "pct_above_2.0": float((aspect > 2.0).mean() * 100),
            "pct_above_2.5": float((aspect > 2.5).mean() * 100),
        },
    }


def plot_morphometric_distributions(dfs: dict[str, pd.DataFrame], output_path: Path):
    """Plot morphometric feature distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for name, df in dfs.items():
        aspect = df["max_diameter_um"] / df["min_diameter_um"].clip(lower=0.01)
        sol = df["solidity"].clip(upper=1.0)

        axes[0].hist(df["area_um2"], bins=50, alpha=0.5, label=name, density=True)
        axes[0].set_xlabel("Cell Area (μm²)")
        axes[0].set_title("Area Distribution")
        axes[0].axvline(25, color="orange", ls="--", label="Moderate threshold")
        axes[0].axvline(80, color="red", ls="--", label="Debate threshold")

        axes[1].hist(sol, bins=50, alpha=0.5, label=name, density=True)
        axes[1].set_xlabel("Solidity")
        axes[1].set_title("Solidity Distribution")
        axes[1].axvline(0.9, color="red", ls="--", label="Threshold")

        axes[2].hist(aspect.clip(upper=6), bins=50, alpha=0.5, label=name, density=True)
        axes[2].set_xlabel("Aspect Ratio")
        axes[2].set_title("Aspect Ratio Distribution")
        axes[2].axvline(2.0, color="red", ls="--", label="Strict threshold")
        axes[2].axvline(2.5, color="orange", ls="--", label="Standard threshold")

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Morphometric Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def apply_recommended_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Apply the recommended filter based on data analysis.

    Uses 'Standard' thresholds: area≥25μm², solidity≥0.9, aspect≤2.5
    These are data-informed: the debate's 80μm² threshold would remove 75%+ of cells.
    """
    name = "Standard (area≥25, sol≥0.9, asp≤2.5)"
    return apply_filter(df, min_area=25, min_solidity=0.9, max_aspect=2.5), name


def main():
    parser = argparse.ArgumentParser(description="D3: Morphometric filtering")
    parser.add_argument("--input-dir", type=Path,
                        default=Path("/tmp/pipeline_comparison/v2_cells_nuclei/cell_data"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  D3: Per-Cell Morphometric Filtering")
    print("=" * 60)

    # Load data
    dfs = {}
    for pq_file in sorted(args.input_dir.glob("*_cells.parquet")):
        slide_name = pq_file.stem.replace("_cells", "")
        df = pd.read_parquet(pq_file)
        dfs[slide_name] = df
        print(f"  Loaded {slide_name}: {len(df)} cells")

    if not dfs:
        print("  ERROR: No parquet files found")
        sys.exit(1)

    # --- Morphometric summary statistics ---
    print("\n--- Morphometric Summary ---")
    all_stats = []
    for name, df in dfs.items():
        stats = morphometric_summary_stats(df, name)
        all_stats.append(stats)
        print(f"\n  {name}:")
        print(f"    Area: mean={stats['area_um2']['mean']:.1f}, median={stats['area_um2']['median']:.1f}, "
              f"p5={stats['area_um2']['p5']:.1f}, p95={stats['area_um2']['p95']:.1f}")
        print(f"    Solidity: mean={stats['solidity']['mean']:.3f}, "
              f"<0.9: {stats['solidity']['pct_below_0.9']:.1f}%, "
              f">1.0: {stats['solidity']['pct_above_1.0']:.1f}%")
        print(f"    Aspect: mean={stats['aspect_ratio']['mean']:.2f}, "
              f">2.0: {stats['aspect_ratio']['pct_above_2.0']:.1f}%, "
              f">2.5: {stats['aspect_ratio']['pct_above_2.5']:.1f}%")

    # --- Threshold sensitivity analysis ---
    print("\n--- Threshold Sensitivity Analysis ---")
    all_results = []
    for name, df in dfs.items():
        results = analyze_threshold_sensitivity(df, name)
        all_results.extend(results)

        print(f"\n  {name}:")
        print(f"  {'Filter':<40} {'Retained':>10} {'H-Score':>10} {'% Pos':>10} {'% 3+':>10}")
        print(f"  {'-'*80}")
        for r in results:
            n3 = r["grade_dist"].get(3, 0)
            pct3 = n3 / r["total_after"] * 100 if r["total_after"] > 0 else 0
            print(f"  {r['filter']:<40} {r['pct_retained']:>9.1f}% {r['h_score']:>10.1f} "
                  f"{r['pct_positive']:>9.1f}% {pct3:>9.1f}%")

    # --- Apply recommended filter and save ---
    print("\n--- Applying Recommended Filter ---")
    for name, df in dfs.items():
        filtered, filter_name = apply_recommended_filter(df)
        print(f"\n  {name}: {len(df)} -> {len(filtered)} cells "
              f"({len(filtered)/len(df)*100:.1f}% retained)")

        grades_before = df["cldn18_composite_grade"].values.astype(int)
        grades_after = filtered["cldn18_composite_grade"].values.astype(int)

        print(f"    H-Score: {compute_hscore(grades_before):.1f} -> {compute_hscore(grades_after):.1f}")
        print(f"    % Positive: {compute_pct_positive(grades_before):.1f}% -> {compute_pct_positive(grades_after):.1f}%")

        # Save filtered parquet
        out_pq = args.output_dir / f"{name}_cells_filtered.parquet"
        filtered.to_parquet(out_pq, index=False)
        print(f"    Saved: {out_pq}")

    # --- Generate plots ---
    plot_sensitivity(all_results, args.output_dir / "threshold_sensitivity.png")
    plot_morphometric_distributions(dfs, args.output_dir / "feature_distributions.png")

    # --- Save results JSON ---
    results_json = {
        "morphometric_stats": all_stats,
        "sensitivity_analysis": [
            {k: v for k, v in r.items() if k != "max_aspect" or v != float("inf")}
            for r in all_results
        ],
        "recommended_filter": "Standard (area≥25, sol≥0.9, asp≤2.5)",
        "note": "Debate threshold (area≥80) would remove 75%+ of cells — too aggressive for our model's polygon sizes. "
                "Standard threshold removes small fragments and elongated fibroblasts while retaining epithelial population.",
        "solidity_warning": "Solidity values >1.0 detected in data — likely numerical artifact from polygon computation. "
                           "Values clamped to 1.0 for filtering.",
    }
    with open(args.output_dir / "morphometric_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    print(f"\n  All results saved to {args.output_dir}/")
    print("  Done.")


if __name__ == "__main__":
    main()
