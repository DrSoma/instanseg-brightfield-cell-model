#!/home/fernandosoto/claudin18_venv/bin/python
"""Cohort eligibility analysis: compare our model vs baseline.

Loads all cohort parquet files (with membrane columns from script 11)
and the baseline region summary CSV, then:
1. For each matching slide: compare cell count, H-score, % positive, grades
2. Determine zolbetuximab eligibility (>=75% cells at >=2+ membrane staining)
3. Identify eligibility "flips" between baseline and our model
4. Generate scatter plots and summary statistics

ALL outputs are flagged: "THRESHOLDS_UNCALIBRATED -- RESEARCH USE ONLY"

Outputs:
  evaluation/eligibility_analysis/cohort_comparison.csv
  evaluation/eligibility_analysis/eligibility_flips.csv
  evaluation/eligibility_analysis/eligibility_summary.json
  evaluation/eligibility_analysis/comparison_plots.png

Usage:
    python scripts/12_eligibility_analysis.py
    python scripts/12_eligibility_analysis.py --cohort-dir /tmp/cohort_v1/cell_data
    python scripts/12_eligibility_analysis.py --baseline /pathodata/.../cldn18_region_summary.csv
    python scripts/12_eligibility_analysis.py --region Patient_Tissue
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Headless environment -- set before any Qt/matplotlib import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("eligibility_analysis")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "eligibility_analysis"

# Eligibility threshold: >=75% of evaluable tumor cells with >=2+ membrane
ELIGIBILITY_PCT_THRESHOLD = 75.0  # percent
ELIGIBILITY_GRADE_THRESHOLD = 2   # >= this grade counts as positive

# Clinical thresholds for composite grade (PRELIMINARY / UNCALIBRATED)
# Must match 11_add_membrane_columns.py
GRADE_THRESHOLDS = (0.10, 0.20, 0.35)

# Default baseline CSV path
DEFAULT_BASELINE_CSV = Path(
    "/pathodata/Claudin18_project/exports/cldn18_region_summary.csv"
)

# Disclaimer stamped on all outputs
DISCLAIMER = (
    "THRESHOLDS_UNCALIBRATED -- RESEARCH USE ONLY. "
    "Do NOT use for clinical decisions. "
    "Composite grades use preliminary thresholds (0.10/0.20/0.35) "
    "that have not been calibrated against pathologist consensus."
)


# ---------------------------------------------------------------------------
# H-score and classification helpers
# ---------------------------------------------------------------------------

def classify_dab(dab_value: float) -> int:
    """Classify DAB OD into 0/1+/2+/3+ using preliminary thresholds."""
    t1, t2, t3 = GRADE_THRESHOLDS
    if np.isnan(dab_value):
        return 0
    if dab_value < t1:
        return 0
    elif dab_value < t2:
        return 1
    elif dab_value < t3:
        return 2
    return 3


def compute_hscore(grades: np.ndarray) -> float:
    """H-score = 1*%1+ + 2*%2+ + 3*%3+. Range 0-300."""
    if len(grades) == 0:
        return 0.0
    n = len(grades)
    return float(
        1 * (grades == 1).sum() / n * 100
        + 2 * (grades == 2).sum() / n * 100
        + 3 * (grades == 3).sum() / n * 100
    )


def compute_pct_positive(grades: np.ndarray) -> float:
    """% cells with >= 2+ membrane staining (moderate or strong)."""
    if len(grades) == 0:
        return 0.0
    return float((grades >= ELIGIBILITY_GRADE_THRESHOLD).sum() / len(grades) * 100)


def is_eligible(pct_positive: float) -> bool:
    """Zolbetuximab eligibility: >= 75% cells at >= 2+."""
    return pct_positive >= ELIGIBILITY_PCT_THRESHOLD


# ---------------------------------------------------------------------------
# Load baseline CSV
# ---------------------------------------------------------------------------

def load_baseline(
    csv_path: Path, target_region: str = "Patient_Tissue"
) -> dict[str, dict[str, Any]]:
    """Load baseline region summary CSV.

    Returns dict keyed by slide_name (e.g. 'CLDN0285') with fields:
      total_cells, negative, weak_1plus, moderate_2plus, strong_3plus,
      pct_positive, h_score, mean_dab_od
    """
    results: dict[str, dict[str, Any]] = {}

    if not csv_path.exists():
        logger.error("Baseline CSV not found: %s", csv_path)
        return results

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide = row.get("slide_name", "").strip()
            region = row.get("region", "").strip()

            # Skip duplicate header rows
            if slide == "slide_name":
                continue

            # Filter to target region
            if region != target_region:
                continue

            try:
                data = {
                    "total_cells": int(row.get("total_cells", 0)),
                    "negative": int(row.get("negative", 0)),
                    "weak_1plus": int(row.get("weak_1plus", 0)),
                    "moderate_2plus": int(row.get("moderate_2plus", 0)),
                    "strong_3plus": int(row.get("strong_3plus", 0)),
                    "pct_positive": float(row.get("pct_positive", 0)),
                    "h_score": float(row.get("h_score", 0)),
                    "mean_dab_od": float(row.get("mean_dab_od", 0)),
                }
            except (ValueError, TypeError) as exc:
                logger.debug("Skipping row %s/%s: %s", slide, region, exc)
                continue

            # Compute pct at >=2+ from the count columns
            tc = data["total_cells"]
            if tc > 0:
                data["pct_2plus"] = (
                    (data["moderate_2plus"] + data["strong_3plus"]) / tc * 100
                )
            else:
                data["pct_2plus"] = 0.0

            data["eligible"] = is_eligible(data["pct_2plus"])
            results[slide] = data

    logger.info(
        "Loaded baseline: %d slides with region=%s", len(results), target_region
    )
    return results


# ---------------------------------------------------------------------------
# Load cohort parquet files
# ---------------------------------------------------------------------------

def load_cohort(
    cohort_dir: Path, target_region: str = "Patient_Tissue"
) -> dict[str, dict[str, Any]]:
    """Load all cohort parquet files and compute per-slide summary.

    Returns dict keyed by slide_id with:
      total_cells, grade_distribution (0/1/2/3 counts),
      pct_positive (>=2+), h_score, mean_membrane_dab,
      mean_completeness, mean_thickness_um, eligible
    """
    import pyarrow.parquet as pq

    results: dict[str, dict[str, Any]] = {}
    parquet_files = sorted(cohort_dir.glob("*_cells.parquet"))

    if not parquet_files:
        logger.error("No parquet files found in %s", cohort_dir)
        return results

    for pf in parquet_files:
        slide_id = pf.stem.replace("_cells", "")

        try:
            table = pq.read_table(pf)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", pf.name, exc)
            continue

        # Check that membrane columns exist
        has_membrane = "membrane_ring_dab" in table.column_names
        has_grades = "cldn18_composite_grade" in table.column_names

        if not has_membrane:
            logger.warning(
                "%s: missing membrane_ring_dab column -- "
                "run 11_add_membrane_columns.py first",
                slide_id,
            )

        # Filter to target region if column exists
        if "region" in table.column_names and target_region:
            regions = table.column("region").to_pylist()
            mask = [r == target_region for r in regions]
            indices = [i for i, m in enumerate(mask) if m]
            if not indices:
                # Try without region filter (some slides may not have it)
                logger.debug(
                    "%s: no rows for region=%s, using all rows",
                    slide_id, target_region,
                )
                indices = list(range(table.num_rows))
            table = table.take(indices)

        n_cells = table.num_rows
        if n_cells == 0:
            continue

        data: dict[str, Any] = {
            "total_cells": n_cells,
            "has_membrane_cols": has_membrane,
        }

        # --- From membrane columns (primary) ---
        if has_grades:
            grades = table.column("cldn18_composite_grade").to_numpy()
            grade_counts = {
                0: int((grades == 0).sum()),
                1: int((grades == 1).sum()),
                2: int((grades == 2).sum()),
                3: int((grades == 3).sum()),
            }
            data["grade_counts"] = grade_counts
            data["negative"] = grade_counts[0]
            data["weak_1plus"] = grade_counts[1]
            data["moderate_2plus"] = grade_counts[2]
            data["strong_3plus"] = grade_counts[3]
            data["pct_2plus"] = compute_pct_positive(grades)
            data["h_score"] = compute_hscore(grades)
            data["eligible"] = is_eligible(data["pct_2plus"])
        elif has_membrane:
            # Compute grades from raw membrane DAB
            membrane_dab = table.column("membrane_ring_dab").to_numpy()
            grades = np.array([classify_dab(v) for v in membrane_dab])
            grade_counts = {
                0: int((grades == 0).sum()),
                1: int((grades == 1).sum()),
                2: int((grades == 2).sum()),
                3: int((grades == 3).sum()),
            }
            data["grade_counts"] = grade_counts
            data["negative"] = grade_counts[0]
            data["weak_1plus"] = grade_counts[1]
            data["moderate_2plus"] = grade_counts[2]
            data["strong_3plus"] = grade_counts[3]
            data["pct_2plus"] = compute_pct_positive(grades)
            data["h_score"] = compute_hscore(grades)
            data["eligible"] = is_eligible(data["pct_2plus"])
        else:
            # No membrane data: mark as unmeasured
            data["grade_counts"] = {0: n_cells, 1: 0, 2: 0, 3: 0}
            data["negative"] = n_cells
            data["weak_1plus"] = 0
            data["moderate_2plus"] = 0
            data["strong_3plus"] = 0
            data["pct_2plus"] = 0.0
            data["h_score"] = 0.0
            data["eligible"] = False

        # --- Raw measurement summary ---
        if has_membrane:
            membrane_dab_arr = table.column("membrane_ring_dab").to_numpy()
            valid_mask = np.isfinite(membrane_dab_arr)
            data["mean_membrane_dab"] = (
                float(membrane_dab_arr[valid_mask].mean())
                if valid_mask.sum() > 0 else 0.0
            )
            data["median_membrane_dab"] = (
                float(np.median(membrane_dab_arr[valid_mask]))
                if valid_mask.sum() > 0 else 0.0
            )
            data["n_measured"] = int(valid_mask.sum())
            data["pct_measured"] = float(valid_mask.sum() / n_cells * 100)

            if "membrane_completeness" in table.column_names:
                comp = table.column("membrane_completeness").to_numpy()
                valid_comp = np.isfinite(comp)
                data["mean_completeness"] = (
                    float(comp[valid_comp].mean())
                    if valid_comp.sum() > 0 else 0.0
                )

            if "membrane_thickness_um" in table.column_names:
                thick = table.column("membrane_thickness_um").to_numpy()
                valid_thick = np.isfinite(thick)
                data["mean_thickness_um"] = (
                    float(thick[valid_thick].mean())
                    if valid_thick.sum() > 0 else 0.0
                )

            if "raw_membrane_dab" in table.column_names:
                raw = table.column("raw_membrane_dab").to_numpy()
                valid_raw = np.isfinite(raw)
                data["mean_raw_dab"] = (
                    float(raw[valid_raw].mean())
                    if valid_raw.sum() > 0 else 0.0
                )
        else:
            data["mean_membrane_dab"] = 0.0
            data["n_measured"] = 0
            data["pct_measured"] = 0.0

        results[slide_id] = data

    logger.info(
        "Loaded cohort: %d slides from %s", len(results), cohort_dir
    )
    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def build_comparison(
    baseline: dict[str, dict[str, Any]],
    cohort: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-slide comparison rows for slides in both datasets."""
    all_slides = sorted(set(baseline.keys()) | set(cohort.keys()))
    rows = []

    for slide_id in all_slides:
        bl = baseline.get(slide_id)
        co = cohort.get(slide_id)

        row: dict[str, Any] = {
            "slide_id": slide_id,
            "in_baseline": bl is not None,
            "in_cohort": co is not None,
        }

        if bl:
            row["bl_total_cells"] = bl["total_cells"]
            row["bl_pct_positive"] = bl.get("pct_positive", 0)
            row["bl_pct_2plus"] = bl.get("pct_2plus", 0)
            row["bl_h_score"] = bl.get("h_score", 0)
            row["bl_mean_dab"] = bl.get("mean_dab_od", 0)
            row["bl_negative"] = bl.get("negative", 0)
            row["bl_weak_1plus"] = bl.get("weak_1plus", 0)
            row["bl_moderate_2plus"] = bl.get("moderate_2plus", 0)
            row["bl_strong_3plus"] = bl.get("strong_3plus", 0)
            row["bl_eligible"] = bl.get("eligible", False)
        else:
            for k in [
                "bl_total_cells", "bl_pct_positive", "bl_pct_2plus",
                "bl_h_score", "bl_mean_dab", "bl_negative", "bl_weak_1plus",
                "bl_moderate_2plus", "bl_strong_3plus",
            ]:
                row[k] = None
            row["bl_eligible"] = None

        if co:
            row["co_total_cells"] = co["total_cells"]
            row["co_pct_2plus"] = co.get("pct_2plus", 0)
            row["co_h_score"] = co.get("h_score", 0)
            row["co_mean_membrane_dab"] = co.get("mean_membrane_dab", 0)
            row["co_mean_raw_dab"] = co.get("mean_raw_dab", 0)
            row["co_mean_completeness"] = co.get("mean_completeness", 0)
            row["co_mean_thickness_um"] = co.get("mean_thickness_um", 0)
            row["co_negative"] = co.get("negative", 0)
            row["co_weak_1plus"] = co.get("weak_1plus", 0)
            row["co_moderate_2plus"] = co.get("moderate_2plus", 0)
            row["co_strong_3plus"] = co.get("strong_3plus", 0)
            row["co_eligible"] = co.get("eligible", False)
            row["co_has_membrane"] = co.get("has_membrane_cols", False)
            row["co_n_measured"] = co.get("n_measured", 0)
            row["co_pct_measured"] = co.get("pct_measured", 0)
        else:
            for k in [
                "co_total_cells", "co_pct_2plus", "co_h_score",
                "co_mean_membrane_dab", "co_mean_raw_dab",
                "co_mean_completeness", "co_mean_thickness_um",
                "co_negative", "co_weak_1plus", "co_moderate_2plus",
                "co_strong_3plus", "co_n_measured", "co_pct_measured",
            ]:
                row[k] = None
            row["co_eligible"] = None
            row["co_has_membrane"] = None

        # Deltas (only if both present)
        if bl and co:
            row["delta_total_cells"] = (
                co["total_cells"] - bl["total_cells"]
            )
            row["delta_pct_2plus"] = (
                co.get("pct_2plus", 0) - bl.get("pct_2plus", 0)
            )
            row["delta_h_score"] = (
                co.get("h_score", 0) - bl.get("h_score", 0)
            )
            # Eligibility flip detection
            bl_elig = bl.get("eligible", False)
            co_elig = co.get("eligible", False)
            if bl_elig and not co_elig:
                row["eligibility_flip"] = "LOST"   # was eligible, now not
            elif not bl_elig and co_elig:
                row["eligibility_flip"] = "GAINED"  # was not eligible, now is
            elif bl_elig and co_elig:
                row["eligibility_flip"] = "BOTH_ELIGIBLE"
            else:
                row["eligibility_flip"] = "BOTH_NOT_ELIGIBLE"
        else:
            row["delta_total_cells"] = None
            row["delta_pct_2plus"] = None
            row["delta_h_score"] = None
            row["eligibility_flip"] = "MISSING_DATA"

        rows.append(row)

    return rows


def identify_flips(comparison: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract slides where eligibility changed between baseline and cohort."""
    return [
        r for r in comparison
        if r.get("eligibility_flip") in ("GAINED", "LOST")
    ]


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def build_summary(
    baseline: dict[str, dict[str, Any]],
    cohort: dict[str, dict[str, Any]],
    comparison: list[dict[str, Any]],
    flips: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build aggregate summary statistics."""
    matched = [
        r for r in comparison
        if r.get("in_baseline") and r.get("in_cohort")
    ]

    # Baseline stats
    bl_eligible = [s for s, d in baseline.items() if d.get("eligible", False)]
    bl_not_eligible = [s for s, d in baseline.items() if not d.get("eligible", False)]

    # Cohort stats
    co_eligible = [s for s, d in cohort.items() if d.get("eligible", False)]
    co_not_eligible = [s for s, d in cohort.items() if not d.get("eligible", False)]
    co_has_membrane = [
        s for s, d in cohort.items() if d.get("has_membrane_cols", False)
    ]

    # Concordance (among matched slides)
    concordant = [
        r for r in matched
        if r.get("eligibility_flip") in ("BOTH_ELIGIBLE", "BOTH_NOT_ELIGIBLE")
    ]
    discordant = [
        r for r in matched
        if r.get("eligibility_flip") in ("GAINED", "LOST")
    ]

    # Cell count and measurement stats from matched slides
    bl_cell_counts = [
        r["bl_total_cells"] for r in matched if r.get("bl_total_cells")
    ]
    co_cell_counts = [
        r["co_total_cells"] for r in matched if r.get("co_total_cells")
    ]

    summary = {
        "disclaimer": DISCLAIMER,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thresholds": {
            "grade_thresholds_dab_od": list(GRADE_THRESHOLDS),
            "eligibility_pct": ELIGIBILITY_PCT_THRESHOLD,
            "eligibility_grade": f">={ELIGIBILITY_GRADE_THRESHOLD}+",
            "calibrated": False,
        },
        "baseline": {
            "source": str(DEFAULT_BASELINE_CSV),
            "n_slides": len(baseline),
            "n_eligible": len(bl_eligible),
            "n_not_eligible": len(bl_not_eligible),
            "eligibility_rate_pct": (
                len(bl_eligible) / max(len(baseline), 1) * 100
            ),
        },
        "cohort": {
            "n_slides": len(cohort),
            "n_with_membrane_cols": len(co_has_membrane),
            "n_eligible": len(co_eligible),
            "n_not_eligible": len(co_not_eligible),
            "eligibility_rate_pct": (
                len(co_eligible) / max(len(cohort), 1) * 100
            ),
            "total_cells": sum(
                d.get("total_cells", 0) for d in cohort.values()
            ),
            "total_measured": sum(
                d.get("n_measured", 0) for d in cohort.values()
            ),
        },
        "comparison": {
            "n_matched_slides": len(matched),
            "n_concordant": len(concordant),
            "n_discordant": len(discordant),
            "concordance_rate_pct": (
                len(concordant) / max(len(matched), 1) * 100
            ),
            "n_gained_eligibility": sum(
                1 for r in flips if r["eligibility_flip"] == "GAINED"
            ),
            "n_lost_eligibility": sum(
                1 for r in flips if r["eligibility_flip"] == "LOST"
            ),
        },
        "cell_counts": {
            "baseline_mean": (
                float(np.mean(bl_cell_counts)) if bl_cell_counts else 0
            ),
            "baseline_median": (
                float(np.median(bl_cell_counts)) if bl_cell_counts else 0
            ),
            "cohort_mean": (
                float(np.mean(co_cell_counts)) if co_cell_counts else 0
            ),
            "cohort_median": (
                float(np.median(co_cell_counts)) if co_cell_counts else 0
            ),
        },
    }

    # Correlation stats if enough matched slides
    if len(matched) >= 3:
        bl_pcts = np.array([
            r["bl_pct_2plus"] for r in matched
            if r.get("bl_pct_2plus") is not None
        ])
        co_pcts = np.array([
            r["co_pct_2plus"] for r in matched
            if r.get("co_pct_2plus") is not None
        ])
        if len(bl_pcts) == len(co_pcts) and len(bl_pcts) >= 3:
            corr = np.corrcoef(bl_pcts, co_pcts)[0, 1]
            mae = float(np.abs(bl_pcts - co_pcts).mean())
            summary["comparison"]["pct_2plus_correlation"] = float(corr)
            summary["comparison"]["pct_2plus_mae"] = mae

        bl_hs = np.array([
            r["bl_h_score"] for r in matched
            if r.get("bl_h_score") is not None
        ])
        co_hs = np.array([
            r["co_h_score"] for r in matched
            if r.get("co_h_score") is not None
        ])
        if len(bl_hs) == len(co_hs) and len(bl_hs) >= 3:
            corr_h = np.corrcoef(bl_hs, co_hs)[0, 1]
            mae_h = float(np.abs(bl_hs - co_hs).mean())
            summary["comparison"]["h_score_correlation"] = float(corr_h)
            summary["comparison"]["h_score_mae"] = mae_h

    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_plots(
    comparison: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate comparison scatter plots."""
    matched = [
        r for r in comparison
        if r.get("in_baseline") and r.get("in_cohort")
    ]

    if not matched:
        logger.warning("No matched slides for plotting")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Cohort vs Baseline Comparison\n"
        f"({len(matched)} matched slides) -- THRESHOLDS UNCALIBRATED",
        fontsize=13, fontweight="bold",
    )

    # --- Plot 1: % at >=2+ (scatter) ---
    ax = axes[0, 0]
    bl_pct = [r["bl_pct_2plus"] for r in matched if r.get("bl_pct_2plus") is not None]
    co_pct = [r["co_pct_2plus"] for r in matched if r.get("co_pct_2plus") is not None]
    if len(bl_pct) == len(co_pct) and len(bl_pct) > 0:
        # Color by eligibility flip
        colors = []
        for r in matched:
            if r.get("bl_pct_2plus") is None or r.get("co_pct_2plus") is None:
                continue
            flip = r.get("eligibility_flip", "")
            if flip == "GAINED":
                colors.append("green")
            elif flip == "LOST":
                colors.append("red")
            elif flip == "BOTH_ELIGIBLE":
                colors.append("blue")
            else:
                colors.append("gray")

        ax.scatter(bl_pct, co_pct, c=colors, alpha=0.6, s=30, edgecolors="k", linewidths=0.3)
        ax.axhline(ELIGIBILITY_PCT_THRESHOLD, color="red", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(ELIGIBILITY_PCT_THRESHOLD, color="red", ls="--", lw=0.8, alpha=0.5)
        lim = max(max(bl_pct + co_pct, default=100) + 5, 105)
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=0.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    ax.set_xlabel("Baseline % at >=2+")
    ax.set_ylabel("Cohort % at >=2+ (UNCALIBRATED)")
    ax.set_title("Eligibility Metric: % at >=2+")

    # --- Plot 2: H-score (scatter) ---
    ax = axes[0, 1]
    bl_hs = [r["bl_h_score"] for r in matched if r.get("bl_h_score") is not None]
    co_hs = [r["co_h_score"] for r in matched if r.get("co_h_score") is not None]
    if len(bl_hs) == len(co_hs) and len(bl_hs) > 0:
        ax.scatter(bl_hs, co_hs, c="steelblue", alpha=0.6, s=30, edgecolors="k", linewidths=0.3)
        lim_h = max(max(bl_hs + co_hs, default=300) + 10, 310)
        ax.plot([0, lim_h], [0, lim_h], "k--", alpha=0.3, lw=0.5)
        ax.set_xlim(0, lim_h)
        ax.set_ylim(0, lim_h)
    ax.set_xlabel("Baseline H-score")
    ax.set_ylabel("Cohort H-score (UNCALIBRATED)")
    ax.set_title("H-score Comparison")

    # --- Plot 3: Cell count ratio ---
    ax = axes[1, 0]
    bl_cells = [r["bl_total_cells"] for r in matched if r.get("bl_total_cells")]
    co_cells = [r["co_total_cells"] for r in matched if r.get("co_total_cells")]
    if len(bl_cells) == len(co_cells) and len(bl_cells) > 0:
        ax.scatter(bl_cells, co_cells, c="darkgreen", alpha=0.6, s=30, edgecolors="k", linewidths=0.3)
        lim_c = max(max(bl_cells + co_cells, default=1e6) * 1.1, 1e5)
        ax.plot([0, lim_c], [0, lim_c], "k--", alpha=0.3, lw=0.5)
        ax.set_xlim(0, lim_c)
        ax.set_ylim(0, lim_c)
    ax.set_xlabel("Baseline Total Cells")
    ax.set_ylabel("Cohort Total Cells")
    ax.set_title("Cell Count Comparison")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

    # --- Plot 4: Eligibility distribution ---
    ax = axes[1, 1]
    flip_counts = {
        "BOTH_ELIGIBLE": 0,
        "BOTH_NOT_ELIGIBLE": 0,
        "GAINED": 0,
        "LOST": 0,
        "MISSING_DATA": 0,
    }
    for r in comparison:
        f = r.get("eligibility_flip", "MISSING_DATA")
        flip_counts[f] = flip_counts.get(f, 0) + 1

    labels = []
    sizes = []
    bar_colors = []
    color_map = {
        "BOTH_ELIGIBLE": "#2196F3",
        "BOTH_NOT_ELIGIBLE": "#9E9E9E",
        "GAINED": "#4CAF50",
        "LOST": "#F44336",
        "MISSING_DATA": "#FFC107",
    }
    for k, v in flip_counts.items():
        if v > 0:
            labels.append(f"{k} ({v})")
            sizes.append(v)
            bar_colors.append(color_map.get(k, "#999999"))

    if sizes:
        ax.barh(labels, sizes, color=bar_colors, edgecolor="k", linewidth=0.5)
        ax.set_xlabel("Number of Slides")
    ax.set_title("Eligibility Status Distribution")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Add disclaimer footer
    fig.text(
        0.5, 0.005, DISCLAIMER,
        ha="center", va="bottom", fontsize=7, style="italic", color="red",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plots saved to %s", output_path)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_comparison_csv(
    rows: list[dict[str, Any]], output_path: Path
) -> None:
    """Write per-slide comparison CSV with disclaimer header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        logger.warning("No comparison rows to write")
        return

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        # Disclaimer as comment line
        f.write(f"# {DISCLAIMER}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Comparison CSV: %s (%d rows)", output_path, len(rows))


def write_flips_csv(
    flips: list[dict[str, Any]], output_path: Path
) -> None:
    """Write eligibility flips CSV with disclaimer header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not flips:
        logger.info("No eligibility flips found")
        # Write empty CSV with header
        with open(output_path, "w", newline="") as f:
            f.write(f"# {DISCLAIMER}\n")
            f.write("# No eligibility flips detected.\n")
        return

    fieldnames = list(flips[0].keys())

    with open(output_path, "w", newline="") as f:
        f.write(f"# {DISCLAIMER}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flips)

    logger.info("Flips CSV: %s (%d flips)", output_path, len(flips))


def write_summary_json(
    summary: dict[str, Any], output_path: Path
) -> None:
    """Write aggregate summary JSON with disclaimer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Summary JSON: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cohort eligibility analysis: compare our model vs baseline. "
                    "ALL outputs marked UNCALIBRATED -- RESEARCH USE ONLY.",
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=Path("/tmp/cohort_v1/cell_data"),
        help="Directory with *_cells.parquet files (default: /tmp/cohort_v1/cell_data)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_CSV,
        help="Baseline region summary CSV "
             f"(default: {DEFAULT_BASELINE_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="Patient_Tissue",
        help="Region to filter on (default: Patient_Tissue). "
             "Use empty string for all regions.",
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

    logger.info("=" * 70)
    logger.info("COHORT ELIGIBILITY ANALYSIS")
    logger.info("  %s", DISCLAIMER)
    logger.info("=" * 70)
    logger.info("Cohort dir:  %s", args.cohort_dir)
    logger.info("Baseline:    %s", args.baseline)
    logger.info("Region:      %s", args.region or "(all)")
    logger.info("Output dir:  %s", args.output_dir)
    logger.info("Eligibility: >=%0.f%% cells at >=%d+ (VENTANA SP265-based)",
                ELIGIBILITY_PCT_THRESHOLD, ELIGIBILITY_GRADE_THRESHOLD)

    # ── Load data ─────────────────────────────────────────────────────────
    target_region = args.region if args.region else None
    baseline = load_baseline(args.baseline, target_region=target_region or "Patient_Tissue")
    cohort = load_cohort(args.cohort_dir, target_region=target_region or "Patient_Tissue")

    if not baseline:
        logger.error("No baseline data loaded. Check CSV path and region filter.")
        sys.exit(1)

    if not cohort:
        logger.error("No cohort data loaded. Check parquet directory.")
        sys.exit(1)

    # Check overlap
    matched_ids = sorted(set(baseline.keys()) & set(cohort.keys()))
    baseline_only = sorted(set(baseline.keys()) - set(cohort.keys()))
    cohort_only = sorted(set(cohort.keys()) - set(baseline.keys()))

    logger.info("Slide overlap:")
    logger.info("  Matched (in both):      %d", len(matched_ids))
    logger.info("  Baseline only:          %d", len(baseline_only))
    logger.info("  Cohort only:            %d", len(cohort_only))

    if not matched_ids:
        logger.warning(
            "No overlapping slides between baseline and cohort. "
            "Will still produce outputs for available data."
        )

    # ── Build comparison ──────────────────────────────────────────────────
    comparison = build_comparison(baseline, cohort)
    flips = identify_flips(comparison)
    summary = build_summary(baseline, cohort, comparison, flips)

    logger.info("Eligibility flips: %d", len(flips))
    for f in flips:
        logger.info(
            "  %s: %s (baseline %.1f%% -> cohort %.1f%%)",
            f["slide_id"],
            f["eligibility_flip"],
            f.get("bl_pct_2plus", 0) or 0,
            f.get("co_pct_2plus", 0) or 0,
        )

    # ── Write outputs ─────────────────────────────────────────────────────
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    write_comparison_csv(comparison, out / "cohort_comparison.csv")
    write_flips_csv(flips, out / "eligibility_flips.csv")
    write_summary_json(summary, out / "eligibility_summary.json")
    generate_plots(comparison, out / "comparison_plots.png")

    # ── Print summary ─────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("  Matched slides:         %d", len(matched_ids))
    logger.info("  Baseline eligible:      %d/%d (%.1f%%)",
                summary["baseline"]["n_eligible"],
                summary["baseline"]["n_slides"],
                summary["baseline"]["eligibility_rate_pct"])
    logger.info("  Cohort eligible:        %d/%d (%.1f%%)",
                summary["cohort"]["n_eligible"],
                summary["cohort"]["n_slides"],
                summary["cohort"]["eligibility_rate_pct"])
    logger.info("  Concordance:            %d/%d (%.1f%%)",
                summary["comparison"]["n_concordant"],
                summary["comparison"]["n_matched_slides"],
                summary["comparison"]["concordance_rate_pct"])
    logger.info("  Eligibility gained:     %d",
                summary["comparison"]["n_gained_eligibility"])
    logger.info("  Eligibility lost:       %d",
                summary["comparison"]["n_lost_eligibility"])
    if "pct_2plus_correlation" in summary["comparison"]:
        logger.info("  %%2+ correlation:        %.3f",
                    summary["comparison"]["pct_2plus_correlation"])
        logger.info("  %%2+ MAE:                %.1f%%",
                    summary["comparison"]["pct_2plus_mae"])
    if "h_score_correlation" in summary["comparison"]:
        logger.info("  H-score correlation:    %.3f",
                    summary["comparison"]["h_score_correlation"])
        logger.info("  H-score MAE:            %.1f",
                    summary["comparison"]["h_score_mae"])
    logger.info("=" * 70)
    logger.info("Outputs:")
    logger.info("  %s", out / "cohort_comparison.csv")
    logger.info("  %s", out / "eligibility_flips.csv")
    logger.info("  %s", out / "eligibility_summary.json")
    logger.info("  %s", out / "comparison_plots.png")
    logger.info("=" * 70)
    logger.info("WARNING: %s", DISCLAIMER)


if __name__ == "__main__":
    main()
