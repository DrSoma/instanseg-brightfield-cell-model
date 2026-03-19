"""Evaluate a trained InstanSeg model on the held-out test split.

Loads the trained model, runs inference on every test tile, and computes
per-instance segmentation metrics (mAP, F1, Precision, Recall, Dice)
separately for nuclei and cells.  Results are saved to evaluation/metrics.json
and a formatted summary is printed to the log.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from instanseg_brightfield.config import get_config_hash, load_config
from instanseg_brightfield.pipeline_state import get_git_sha

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IoU / matching helpers
# ---------------------------------------------------------------------------

def _instance_iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute the IoU matrix between every predicted and ground-truth instance.

    Args:
        pred: Integer-labelled prediction mask (H, W).
        gt: Integer-labelled ground-truth mask (H, W).

    Returns:
        IoU matrix of shape (n_pred, n_gt).
    """
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]

    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float64)

    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float64)
    for i, pid in enumerate(pred_ids):
        pred_mask = pred == pid
        for j, gid in enumerate(gt_ids):
            gt_mask = gt == gid
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            if intersection == 0:
                continue
            union = np.logical_and(pred_mask | gt_mask, True).sum()
            iou_matrix[i, j] = intersection / union

    return iou_matrix


def _match_instances(
    iou_matrix: np.ndarray,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Greedy matching of predicted to ground-truth instances.

    Each prediction is assigned to the ground-truth instance with the highest
    IoU (above *iou_threshold*) that has not already been matched.

    Args:
        iou_matrix: Shape (n_pred, n_gt).
        iou_threshold: Minimum IoU for a match.

    Returns:
        (true_positives, false_positives, false_negatives).
    """
    n_pred, n_gt = iou_matrix.shape
    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0

    matched_gt: set[int] = set()
    tp = 0

    # Sort predictions by their best IoU (descending) for stable matching
    best_iou_per_pred = iou_matrix.max(axis=1)
    pred_order = np.argsort(-best_iou_per_pred)

    for pi in pred_order:
        remaining = [j for j in range(n_gt) if j not in matched_gt]
        if not remaining:
            break
        best_j = max(remaining, key=lambda j: iou_matrix[pi, j])
        if iou_matrix[pi, best_j] >= iou_threshold:
            tp += 1
            matched_gt.add(best_j)

    fp = n_pred - tp
    fn = n_gt - len(matched_gt)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 when the denominator is zero."""
    return numerator / denominator if denominator > 0 else 0.0


def _compute_average_precision(
    pred: np.ndarray,
    gt: np.ndarray,
    iou_threshold: float,
) -> float:
    """Compute average precision at a single IoU threshold.

    Uses ``instanseg.utils.metrics._robust_average_precision`` when available,
    otherwise falls back to the greedy-matching implementation above.

    Args:
        pred: Integer-labelled prediction mask (H, W).
        gt: Integer-labelled ground-truth mask (H, W).
        iou_threshold: IoU threshold for positive matches.

    Returns:
        Average precision score in [0.0, 1.0].
    """
    try:
        from instanseg.utils.metrics import _robust_average_precision
        return float(_robust_average_precision(gt, pred, threshold=iou_threshold))
    except (ImportError, AttributeError):
        pass

    iou_mat = _instance_iou_matrix(pred, gt)
    tp, fp, fn = _match_instances(iou_mat, iou_threshold)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    # AP at a single threshold simplifies to precision * recall for the
    # matched set; when there is exactly one threshold the AP is equivalent
    # to the F1-like product weighted by recall.
    return _safe_div(precision * recall, 1.0) if (tp + fp + fn) > 0 else 0.0


def _compute_dice_per_instance(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute mean Dice coefficient over matched instances.

    Matching uses IoU >= 0.5 (greedy).

    Args:
        pred: Integer-labelled prediction mask (H, W).
        gt: Integer-labelled ground-truth mask (H, W).

    Returns:
        Mean Dice coefficient across matched pairs.
    """
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]

    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return 0.0

    iou_mat = _instance_iou_matrix(pred, gt)
    matched_gt: set[int] = set()
    dice_scores: list[float] = []

    best_iou_per_pred = iou_mat.max(axis=1)
    pred_order = np.argsort(-best_iou_per_pred)

    for pi in pred_order:
        remaining = [j for j in range(len(gt_ids)) if j not in matched_gt]
        if not remaining:
            break
        best_j = max(remaining, key=lambda j: iou_mat[pi, j])
        if iou_mat[pi, best_j] >= 0.5:
            matched_gt.add(best_j)
            pred_mask = (pred == pred_ids[pi]).astype(np.float64)
            gt_mask = (gt == gt_ids[best_j]).astype(np.float64)
            intersection = (pred_mask * gt_mask).sum()
            dice = 2.0 * intersection / (pred_mask.sum() + gt_mask.sum())
            dice_scores.append(dice)

    return float(np.mean(dice_scores)) if dice_scores else 0.0


def _compute_cell_nucleus_ratio_stats(
    pred_nuclei: np.ndarray,
    pred_cells: np.ndarray,
) -> dict[str, float]:
    """Compute cell/nucleus area ratio distribution statistics.

    Args:
        pred_nuclei: Predicted nucleus labels (H, W).
        pred_cells: Predicted cell labels (H, W).

    Returns:
        Dictionary with mean, median, std, min, max of the ratio distribution.
    """
    cell_ids = np.unique(pred_cells)
    cell_ids = cell_ids[cell_ids > 0]

    ratios: list[float] = []
    for cid in cell_ids:
        cell_area = float((pred_cells == cid).sum())
        nuc_area = float((pred_nuclei == cid).sum())
        if nuc_area > 0:
            ratios.append(cell_area / nuc_area)

    if not ratios:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    arr = np.array(ratios)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def evaluate_tile(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    iou_thresholds: list[float],
) -> dict[str, Any]:
    """Evaluate a single tile's predictions against ground truth.

    Args:
        pred_labels: Integer-labelled prediction mask (H, W).
        gt_labels: Integer-labelled ground-truth mask (H, W).
        iou_thresholds: List of IoU thresholds for mAP and detection metrics.

    Returns:
        Dictionary containing all per-tile metrics.
    """
    metrics: dict[str, Any] = {
        "num_pred": int(len(np.unique(pred_labels)) - (1 if 0 in pred_labels else 0)),
        "num_gt": int(len(np.unique(gt_labels)) - (1 if 0 in gt_labels else 0)),
    }

    # mAP at each IoU threshold
    for thr in iou_thresholds:
        thr_key = str(thr).replace(".", "")
        metrics[f"ap_iou{thr_key}"] = _compute_average_precision(pred_labels, gt_labels, thr)

    # Detection metrics at IoU 0.5
    iou_mat = _instance_iou_matrix(pred_labels, gt_labels)
    tp, fp, fn = _match_instances(iou_mat, iou_threshold=0.5)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    metrics["precision_05"] = precision
    metrics["recall_05"] = recall
    metrics["f1_05"] = f1

    # Dice
    metrics["dice"] = _compute_dice_per_instance(pred_labels, gt_labels)

    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_metrics(
    tile_metrics: list[dict[str, Any]],
) -> dict[str, float]:
    """Average per-tile metrics into a single summary dictionary.

    Args:
        tile_metrics: List of per-tile metric dictionaries.

    Returns:
        Dictionary of mean values for every metric key.
    """
    if not tile_metrics:
        return {}

    keys = [k for k in tile_metrics[0] if isinstance(tile_metrics[0][k], (int, float))]
    summary: dict[str, float] = {}
    for key in keys:
        values = [m[key] for m in tile_metrics if key in m]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else 0.0

    return summary


def _format_summary_table(
    nuclei_summary: dict[str, float],
    cells_summary: dict[str, float],
) -> str:
    """Build a formatted summary table for logging.

    Args:
        nuclei_summary: Aggregated nuclei metrics.
        cells_summary: Aggregated cells metrics.

    Returns:
        Multi-line string with the formatted table.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("+" + "-" * 36 + "+" + "-" * 14 + "+" + "-" * 14 + "+")
    lines.append(f"| {'Metric':<34} | {'Nuclei':>12} | {'Cells':>12} |")
    lines.append("+" + "-" * 36 + "+" + "-" * 14 + "+" + "-" * 14 + "+")

    # Align keys by removing the 'mean_' prefix for display
    display_keys = sorted(set(list(nuclei_summary.keys()) + list(cells_summary.keys())))
    for key in display_keys:
        label = key.replace("mean_", "")
        nuc_val = nuclei_summary.get(key, float("nan"))
        cell_val = cells_summary.get(key, float("nan"))
        lines.append(f"| {label:<34} | {nuc_val:>12.4f} | {cell_val:>12.4f} |")

    lines.append("+" + "-" * 36 + "+" + "-" * 14 + "+" + "-" * 14 + "+")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained InstanSeg model on the test split.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to exported InstanSeg model (folder or .pt file). "
             "Defaults to models/<experiment_name>.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda).",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Evaluate on at most N test tiles (for quick sanity checks).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config(args.config)
    cfg_hash = get_config_hash(cfg)
    git_sha = get_git_sha()

    data_dir = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    eval_dir = Path(cfg["paths"]["evaluation_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    experiment_name: str = cfg["training"]["experiment_name"]
    iou_thresholds: list[float] = cfg["evaluation"]["iou_thresholds"]

    logger.info("=" * 72)
    logger.info("InstanSeg Brightfield Evaluation")
    logger.info("=" * 72)
    logger.info("Config hash : %s", cfg_hash)
    logger.info("Git SHA     : %s", git_sha)
    logger.info("Experiment  : %s", experiment_name)
    logger.info("Device      : %s", args.device)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    dataset_path = data_dir / "segmentation_dataset.pth"
    if not dataset_path.exists():
        logger.error("Dataset file not found: %s", dataset_path)
        sys.exit(1)

    logger.info("Loading dataset from %s ...", dataset_path)
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)

    # The dataset is expected to contain a 'test' key (or similar split info).
    # Adapt to the actual structure produced by the dataset-building script.
    if isinstance(dataset, dict) and "Test" in dataset:
        test_data = dataset["Test"]
    elif isinstance(dataset, dict) and "test" in dataset:
        test_data = dataset["test"]
    elif isinstance(dataset, dict) and "split" in dataset:
        test_data = [s for s in dataset["samples"] if s.get("split") == "test"]
    elif isinstance(dataset, list):
        # Fall back: assume the last test_fraction of the list is the test set
        test_frac = cfg["dataset"]["test_fraction"]
        n_test = max(1, int(len(dataset) * test_frac))
        test_data = dataset[-n_test:]
    else:
        # Treat the whole dataset as the evaluation set
        test_data = dataset
        logger.warning("Could not determine test split; evaluating on entire dataset.")

    if args.max_tiles is not None:
        test_data = test_data[: args.max_tiles]

    logger.info("Test tiles: %d", len(test_data))

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_path = args.model_path if args.model_path else str(model_dir / experiment_name)

    logger.info("Loading model from %s ...", model_path)
    try:
        from instanseg import InstanSeg
        model = InstanSeg(model_path, device=args.device)
    except ImportError:
        logger.error(
            "Could not import instanseg. "
            "Install the package: pip install instanseg-torch",
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to load model from %s: %s", model_path, exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    logger.info("Running inference on test tiles ...")
    wall_start = time.monotonic()

    nuclei_metrics: list[dict[str, Any]] = []
    cells_metrics: list[dict[str, Any]] = []
    ratio_stats_all: list[dict[str, float]] = []

    for idx, sample in enumerate(test_data):
        # Extract image and ground truth from the sample.  The exact field
        # names depend on how the dataset was built; we handle common layouts.
        if isinstance(sample, dict):
            image = sample.get("image", sample.get("img"))
            gt_nuclei = sample.get("nucleus_masks", sample.get("nuclei", sample.get("labels_nuclei")))
            gt_cells = sample.get("cell_masks", sample.get("cells", sample.get("labels_cells")))
        elif isinstance(sample, (tuple, list)) and len(sample) >= 3:
            image, gt_nuclei, gt_cells = sample[0], sample[1], sample[2]
        else:
            logger.warning("Skipping sample %d: unrecognised format.", idx)
            continue

        if image is None or gt_nuclei is None:
            logger.warning("Skipping sample %d: missing image or labels.", idx)
            continue

        # Ensure tensors are numpy arrays for metric computation
        if isinstance(image, torch.Tensor):
            image_np: np.ndarray = image.cpu().numpy()
        else:
            image_np = np.asarray(image)

        gt_nuclei_np = gt_nuclei.cpu().numpy() if isinstance(gt_nuclei, torch.Tensor) else np.asarray(gt_nuclei)

        # Run inference -- InstanSeg returns (nuclei_labels, cell_labels) for NC models
        try:
            result = model(image_np)
            if isinstance(result, tuple) and len(result) >= 2:
                pred_nuclei_np, pred_cells_np = (
                    np.asarray(result[0].cpu() if isinstance(result[0], torch.Tensor) else result[0]),
                    np.asarray(result[1].cpu() if isinstance(result[1], torch.Tensor) else result[1]),
                )
            else:
                pred_labels = np.asarray(
                    result.cpu() if isinstance(result, torch.Tensor) else result
                )
                # Single output -- treat as nuclei only
                pred_nuclei_np = pred_labels
                pred_cells_np = None
        except Exception as exc:
            logger.warning("Inference failed on sample %d: %s", idx, exc)
            continue

        # Nuclei metrics
        nuc_m = evaluate_tile(pred_nuclei_np, gt_nuclei_np, iou_thresholds)
        nuclei_metrics.append(nuc_m)

        # Cell metrics (only if both prediction and GT are available)
        if pred_cells_np is not None and gt_cells is not None:
            gt_cells_np = gt_cells.cpu().numpy() if isinstance(gt_cells, torch.Tensor) else np.asarray(gt_cells)
            cell_m = evaluate_tile(pred_cells_np, gt_cells_np, iou_thresholds)
            cells_metrics.append(cell_m)

            # Cell/nucleus ratio distribution
            ratio_stats = _compute_cell_nucleus_ratio_stats(pred_nuclei_np, pred_cells_np)
            ratio_stats_all.append(ratio_stats)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_data):
            logger.info("  Processed %d / %d tiles", idx + 1, len(test_data))

    wall_elapsed = time.monotonic() - wall_start
    logger.info("Inference completed in %.1f seconds.", wall_elapsed)

    # ------------------------------------------------------------------
    # Aggregate and save
    # ------------------------------------------------------------------
    nuclei_summary = _aggregate_metrics(nuclei_metrics)
    cells_summary = _aggregate_metrics(cells_metrics)

    # Aggregate ratio statistics
    ratio_summary: dict[str, float] = {}
    if ratio_stats_all:
        for key in ratio_stats_all[0]:
            vals = [r[key] for r in ratio_stats_all]
            ratio_summary[f"cell_nucleus_ratio_{key}"] = float(np.mean(vals))

    results: dict[str, Any] = {
        "config_hash": cfg_hash,
        "git_sha": git_sha,
        "experiment_name": experiment_name,
        "num_test_tiles": len(test_data),
        "iou_thresholds": iou_thresholds,
        "nuclei": nuclei_summary,
        "cells": cells_summary,
        "cell_nucleus_ratio": ratio_summary,
    }

    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Metrics saved to %s", metrics_path)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    table = _format_summary_table(nuclei_summary, cells_summary)
    logger.info(table)

    if ratio_summary:
        logger.info("Cell/nucleus area ratio (across test set):")
        for key, val in sorted(ratio_summary.items()):
            logger.info("  %-40s %.4f", key, val)

    logger.info("=" * 72)
    logger.info("Evaluation complete.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
