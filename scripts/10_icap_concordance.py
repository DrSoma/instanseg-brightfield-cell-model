#!/usr/bin/env python3
"""D4: iCAP Paired Concordance Test.

Compare model performance on different tissue regions (Patient_Tissue vs iCAP_Control)
on the SAME slide. Also compare with baseline pipeline scores.

The definitive selectivity metric: do positive regions score higher than
expected-negative regions on the same slide?

Usage:
    python scripts/10_icap_concordance.py [--our-dir /tmp/pipeline_comparison/v2_cells_nuclei/cell_data]
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
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "icap_concordance"

# Baseline results
BASELINE_CSV = Path("/pathodata/Claudin18_project/exports/cldn18_region_summary.csv")
REGION_ANN_DIR = Path("/pathodata/Claudin18_project/preprocessing/region_annotations")


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


def load_region_annotations(geojson_path: Path) -> dict:
    """Load region annotations from GeoJSON. Returns {region_name: shapely.Polygon}."""
    from shapely.geometry import shape as shapely_shape

    with open(geojson_path) as f:
        data = json.load(f)

    regions = {}
    for feat in data.get("features", []):
        cls_name = feat.get("properties", {}).get("classification", {}).get("name", "Unknown")
        geom = shapely_shape(feat["geometry"])
        regions[cls_name] = geom

    return regions


def assign_cells_to_regions(df: pd.DataFrame, regions: dict) -> pd.DataFrame:
    """Assign each cell to a region based on centroid containment."""
    from shapely.geometry import Point

    region_labels = []
    for _, row in df.iterrows():
        pt = Point(row["centroid_x"], row["centroid_y"])
        assigned = "Outside"
        for region_name, region_poly in regions.items():
            if region_poly.contains(pt):
                assigned = region_name
                break
        region_labels.append(assigned)

    df = df.copy()
    df["region_assigned"] = region_labels
    return df


def per_region_stats(df: pd.DataFrame, slide_name: str) -> list[dict]:
    """Compute per-region statistics."""
    results = []

    for region in df["region_assigned"].unique():
        region_df = df[df["region_assigned"] == region]
        grades = region_df["cldn18_composite_grade"].values.astype(int)

        n = len(region_df)
        if n == 0:
            continue

        results.append({
            "slide": slide_name,
            "region": region,
            "n_cells": n,
            "h_score": compute_hscore(grades),
            "pct_positive": compute_pct_positive(grades),
            "pct_3plus": float((grades == 3).sum() / n * 100),
            "mean_membrane_dab": float(region_df["membrane_ring_dab"].mean()) if "membrane_ring_dab" in region_df.columns else None,
            "mean_completeness": float(region_df["membrane_completeness"].mean()) if "membrane_completeness" in region_df.columns else None,
            "mean_thickness_px": float(region_df["membrane_thickness_px"].mean()) if "membrane_thickness_px" in region_df.columns else None,
        })

    return results


def load_baseline_results() -> pd.DataFrame:
    """Load baseline pipeline results for comparison."""
    if not BASELINE_CSV.exists():
        return pd.DataFrame()

    # CSV has duplicate header row and some lines have extra fields
    df = pd.read_csv(BASELINE_CSV, skiprows=[1], on_bad_lines="skip")
    return df


def concordance_analysis(our_results: list[dict], baseline_df: pd.DataFrame) -> dict:
    """Compare our model vs baseline across regions."""
    analysis = {
        "slides_analyzed": 0,
        "per_slide": [],
        "aggregate": {},
    }

    our_df = pd.DataFrame(our_results)
    slides = our_df["slide"].unique()
    analysis["slides_analyzed"] = len(slides)

    for slide in slides:
        slide_our = our_df[our_df["slide"] == slide]

        # Our model: Patient_Tissue vs iCAP_Control
        pt_data = slide_our[slide_our["region"] == "Patient_Tissue"]
        icap_data = slide_our[slide_our["region"] == "iCAP_Control"]

        slide_result = {
            "slide": slide,
            "our_model": {},
            "baseline": {},
            "concordance": {},
        }

        if len(pt_data) > 0:
            slide_result["our_model"]["Patient_Tissue"] = pt_data.iloc[0].to_dict()
        if len(icap_data) > 0:
            slide_result["our_model"]["iCAP_Control"] = icap_data.iloc[0].to_dict()

        # Concordance: do both regions agree on expression level?
        if len(pt_data) > 0 and len(icap_data) > 0:
            pt_hscore = pt_data.iloc[0]["h_score"]
            icap_hscore = icap_data.iloc[0]["h_score"]
            pt_pct = pt_data.iloc[0]["pct_positive"]
            icap_pct = icap_data.iloc[0]["pct_positive"]

            slide_result["concordance"] = {
                "h_score_diff": float(pt_hscore - icap_hscore),
                "pct_pos_diff": float(pt_pct - icap_pct),
                "h_score_ratio": float(pt_hscore / icap_hscore) if icap_hscore > 0 else float("inf"),
            }

        # Baseline comparison
        if len(baseline_df) > 0:
            slide_baseline = baseline_df[baseline_df["slide_name"] == slide]
            for _, brow in slide_baseline.iterrows():
                region = brow.get("region", "")
                slide_result["baseline"][region] = {
                    "n_cells": int(brow.get("total_cells", 0)),
                    "h_score": float(brow.get("h_score", 0)),
                    "pct_positive": float(brow.get("pct_positive", 0)),
                }

        analysis["per_slide"].append(slide_result)

    return analysis


def run_on_existing_data(our_dir: Path) -> list[dict]:
    """Run analysis on existing parquet data using the 'region' column if present,
    or re-assign using GeoJSON annotations."""
    all_results = []

    for pq_file in sorted(our_dir.glob("*_cells.parquet")):
        slide_name = pq_file.stem.replace("_cells", "")
        df = pd.read_parquet(pq_file)
        print(f"\n  Processing {slide_name}: {len(df)} cells")

        # Check if region column already exists and is usable
        if "region" in df.columns and df["region"].nunique() > 1:
            print(f"    Using existing 'region' column: {df['region'].unique()}")
            df = df.rename(columns={"region": "region_assigned"})
        else:
            # Load region annotations
            ann_path = REGION_ANN_DIR / f"{slide_name}_regions.geojson"
            if not ann_path.exists():
                print(f"    No region annotations found, skipping")
                continue

            print(f"    Loading region annotations: {ann_path}")
            regions = load_region_annotations(ann_path)
            print(f"    Regions: {list(regions.keys())}")

            # For speed, use vectorized containment
            from shapely.geometry import Point
            from shapely import vectorized

            # Assign using prepared geometries for speed
            df = df.copy()
            df["region_assigned"] = "Outside"

            for region_name, region_poly in regions.items():
                from shapely.prepared import prep
                prepared = prep(region_poly)
                mask = np.array([
                    prepared.contains(Point(x, y))
                    for x, y in zip(df["centroid_x"].values, df["centroid_y"].values)
                ])
                df.loc[mask & (df["region_assigned"] == "Outside"), "region_assigned"] = region_name
                n_in = mask.sum()
                print(f"    {region_name}: {n_in} cells ({n_in/len(df)*100:.1f}%)")

        # Compute stats
        results = per_region_stats(df, slide_name)
        all_results.extend(results)

        # Print summary
        for r in results:
            print(f"    {r['region']}: {r['n_cells']} cells, "
                  f"H={r['h_score']:.0f}, %pos={r['pct_positive']:.1f}%, "
                  f"%3+={r['pct_3plus']:.1f}%")

    return all_results


def plot_concordance(analysis: dict, output_path: Path):
    """Plot concordance analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    slides = []
    pt_hscores = []
    icap_hscores = []
    baseline_pt_hscores = []
    baseline_icap_hscores = []

    for slide_data in analysis["per_slide"]:
        slide = slide_data["slide"]
        slides.append(slide)

        if "Patient_Tissue" in slide_data["our_model"]:
            pt_hscores.append(slide_data["our_model"]["Patient_Tissue"]["h_score"])
        else:
            pt_hscores.append(0)

        if "iCAP_Control" in slide_data["our_model"]:
            icap_hscores.append(slide_data["our_model"]["iCAP_Control"]["h_score"])
        else:
            icap_hscores.append(0)

        if "Patient_Tissue" in slide_data.get("baseline", {}):
            baseline_pt_hscores.append(slide_data["baseline"]["Patient_Tissue"]["h_score"])
        else:
            baseline_pt_hscores.append(0)

        if "iCAP_Control" in slide_data.get("baseline", {}):
            baseline_icap_hscores.append(slide_data["baseline"]["iCAP_Control"]["h_score"])
        else:
            baseline_icap_hscores.append(0)

    x = np.arange(len(slides))
    width = 0.35

    # Our model: PT vs iCAP
    axes[0].bar(x - width/2, pt_hscores, width, label="Patient_Tissue", color="steelblue")
    axes[0].bar(x + width/2, icap_hscores, width, label="iCAP_Control", color="coral")
    axes[0].set_title("Our Model: H-Score by Region")
    axes[0].set_ylabel("H-Score")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(slides, rotation=30, ha="right", fontsize=8)
    axes[0].legend()

    # Baseline: PT vs iCAP
    if any(v > 0 for v in baseline_pt_hscores):
        axes[1].bar(x - width/2, baseline_pt_hscores, width, label="Patient_Tissue", color="steelblue", alpha=0.5)
        axes[1].bar(x + width/2, baseline_icap_hscores, width, label="iCAP_Control", color="coral", alpha=0.5)
        axes[1].set_title("Baseline Pipeline: H-Score by Region")
        axes[1].set_ylabel("H-Score")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(slides, rotation=30, ha="right", fontsize=8)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No baseline data", transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=14)

    # Our model vs baseline (Patient_Tissue)
    if any(v > 0 for v in baseline_pt_hscores):
        axes[2].scatter(baseline_pt_hscores, pt_hscores, s=60, c="steelblue",
                        edgecolors="black", zorder=5)
        axes[2].plot([0, 300], [0, 300], "k--", alpha=0.3, label="y=x")
        axes[2].set_xlabel("Baseline H-Score")
        axes[2].set_ylabel("Our Model H-Score")
        axes[2].set_title("Our Model vs Baseline (Patient_Tissue)")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No baseline data", transform=axes[2].transAxes,
                     ha="center", va="center", fontsize=14)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.suptitle("iCAP Paired Concordance Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="D4: iCAP paired concordance")
    parser.add_argument("--our-dir", type=Path,
                        default=Path("/tmp/pipeline_comparison/v2_cells_nuclei/cell_data"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  D4: iCAP Paired Concordance Test")
    print("=" * 60)

    # Run analysis on existing data
    our_results = run_on_existing_data(args.our_dir)

    if not our_results:
        print("\n  ERROR: No results computed")
        sys.exit(1)

    # Load baseline
    print("\n  Loading baseline results...")
    baseline_df = load_baseline_results()
    print(f"  Baseline: {len(baseline_df)} rows")

    # Concordance analysis
    print("\n--- Concordance Analysis ---")
    analysis = concordance_analysis(our_results, baseline_df)

    for slide_data in analysis["per_slide"]:
        slide = slide_data["slide"]
        print(f"\n  {slide}:")

        if "concordance" in slide_data and slide_data["concordance"]:
            conc = slide_data["concordance"]
            print(f"    H-Score: PT={slide_data['our_model'].get('Patient_Tissue',{}).get('h_score',0):.0f} "
                  f"vs iCAP={slide_data['our_model'].get('iCAP_Control',{}).get('h_score',0):.0f} "
                  f"(diff={conc['h_score_diff']:.0f})")

        for region_name in ["Patient_Tissue", "iCAP_Control"]:
            our = slide_data["our_model"].get(region_name, {})
            base = slide_data["baseline"].get(region_name, {})
            if our:
                our_h = our.get("h_score", 0)
                base_h = base.get("h_score", 0)
                our_n = our.get("n_cells", 0)
                base_n = base.get("n_cells", 0)
                print(f"    {region_name}: Our H={our_h:.0f} ({our_n} cells) vs "
                      f"Baseline H={base_h:.0f} ({base_n} cells)")

    # Generate plot
    plot_concordance(analysis, args.output_dir / "icap_concordance.png")

    # Save results
    results_json = {
        "analysis": analysis,
        "our_per_region": our_results,
        "summary": {
            "slides_analyzed": analysis["slides_analyzed"],
            "regions_found": list(set(r["region"] for r in our_results)),
        },
    }
    with open(args.output_dir / "icap_concordance_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    print(f"\n  Results saved to {args.output_dir}/")
    print("  Done.")


if __name__ == "__main__":
    main()
