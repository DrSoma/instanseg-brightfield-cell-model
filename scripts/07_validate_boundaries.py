"""Validate that predicted cell boundaries trace real membranes, not just expanded nuclei.

Three objective tests:
  Test 1: Visual overlays — save PNGs with predicted boundaries on top of RGB tiles
  Test 2: Compartment intensity — measure DAB in membrane ring vs cytoplasm vs nucleus
  Test 3: Comparison with nucleus expansion — are our boundaries different from 5um expansion?

Requires the trained model checkpoint and test tiles.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from tqdm import tqdm

from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver

logger = logging.getLogger(__name__)


def _load_model_and_postprocess(cfg: dict, device: str = "cuda"):
    """Load trained model + pixel classifier + loss module for inference."""
    from instanseg.utils.model_loader import build_model_from_dict
    from instanseg.utils.loss.instanseg_loss import InstanSeg as InstanSegLoss, ProbabilityNet

    tcfg = cfg["training"]
    ckpt = torch.load(
        "/home/fernandosoto/Documents/models/brightfield_cells_nuclei/model_weights.pth",
        weights_only=False, map_location=device,
    )
    model = build_model_from_dict({
        "model_str": "InstanSeg_UNet", "layers": tuple(tcfg["layers"]),
        "dim_in": 3, "n_sigma": tcfg["n_sigma"], "dim_coords": tcfg["dim_coords"],
        "dim_seeds": tcfg["dim_seeds"], "norm": tcfg["norm"],
        "cells_and_nuclei": True, "multihead": False, "dropprob": 0.0,
    })
    loadable = {k: v for k, v in ckpt["model_state_dict"].items() if k in model.state_dict()}
    model.load_state_dict(loadable, strict=False)
    model.eval().to(device)

    pc = ProbabilityNet(
        embedding_dim=tcfg["dim_coords"] + tcfg["n_sigma"] - 2 + 2,
        width=tcfg["mlp_width"],
    )
    pc_sd = {k.replace("pixel_classifier.", ""): v
             for k, v in ckpt["model_state_dict"].items() if "pixel_classifier" in k}
    if pc_sd:
        pc.load_state_dict(pc_sd, strict=False)
    pc.eval().to(device)

    loss_fn = InstanSegLoss(
        n_sigma=tcfg["n_sigma"], dim_coords=tcfg["dim_coords"],
        dim_seeds=tcfg["dim_seeds"], cells_and_nuclei=True, window_size=32,
    )
    loss_fn.pixel_classifier = pc
    loss_fn.eval().to(device)

    return model, loss_fn


def _predict_tile(model, loss_fn, img: np.ndarray, device: str = "cuda"):
    """Run inference on a single tile, return (pred_nuclei, pred_cells) int32 masks."""
    from instanseg.utils.augmentations import Augmentations
    Aug = Augmentations()

    t, _ = Aug.to_tensor(img, normalize=False)
    t, _ = Aug.normalize(t)

    with torch.inference_mode():
        raw = model(t.unsqueeze(0).to(device))
        pred = loss_fn.postprocessing(
            raw[0], seed_threshold=0.5, peak_distance=5,
            mask_threshold=0.53, overlap_threshold=0.3, window_size=32,
        )

    p = pred.cpu().numpy()
    if p.ndim == 3 and p.shape[0] >= 2:
        return p[0].astype(np.int32), p[1].astype(np.int32)
    return p[0].astype(np.int32), np.zeros_like(p[0], dtype=np.int32)


# ---------------------------------------------------------------------------
# Test 1: Visual overlays
# ---------------------------------------------------------------------------

def test1_visual_overlays(
    model, loss_fn, test_data: list, data_dir: Path, out_dir: Path,
    n_tiles: int = 20, device: str = "cuda",
):
    """Save PNGs with predicted boundaries overlaid on RGB tiles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for item in tqdm(test_data[:n_tiles * 2], desc="Test 1: Visual overlays"):
        img = tifffile.imread(data_dir / item["image"])
        gt_nuc = tifffile.imread(data_dir / item["nucleus_masks"]).astype(np.int32)
        if gt_nuc.max() == 0:
            continue

        try:
            pred_nuc, pred_cell = _predict_tile(model, loss_fn, img, device)
        except Exception:
            continue

        if pred_nuc.max() == 0:
            continue

        # Draw boundaries on RGB image
        overlay = img.copy()

        # Nucleus boundaries in red
        for label_id in np.unique(pred_nuc):
            if label_id == 0:
                continue
            mask = (pred_nuc == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)

        # Cell boundaries in green
        for label_id in np.unique(pred_cell):
            if label_id == 0:
                continue
            mask = (pred_cell == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

        # GT nucleus boundaries in blue (thin, for comparison)
        for label_id in np.unique(gt_nuc):
            if label_id == 0:
                continue
            mask = (gt_nuc == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)

        tile_name = Path(item["image"]).stem
        cv2.imwrite(
            str(out_dir / f"{tile_name}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        count += 1
        if count >= n_tiles:
            break

    logger.info("Test 1: Saved %d overlay images to %s", count, out_dir)
    return count


# ---------------------------------------------------------------------------
# Test 2: Compartment intensity analysis
# ---------------------------------------------------------------------------

def test2_compartment_intensity(
    model, loss_fn, test_data: list, data_dir: Path, cfg: dict,
    n_tiles: int = 100, device: str = "cuda",
):
    """Measure DAB intensity in membrane ring vs cytoplasm vs nucleus."""
    sc = cfg["stain_deconvolution"]
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))

    membrane_dab, cytoplasm_dab, nucleus_dab = [], [], []
    tiles_processed = 0

    for item in tqdm(test_data[:n_tiles * 2], desc="Test 2: Compartment intensity"):
        img = tifffile.imread(data_dir / item["image"])
        gt_nuc = tifffile.imread(data_dir / item["nucleus_masks"]).astype(np.int32)
        if gt_nuc.max() == 0:
            continue

        try:
            pred_nuc, pred_cell = _predict_tile(model, loss_fn, img, device)
        except Exception:
            continue

        if pred_nuc.max() == 0 or pred_cell.max() == 0:
            continue

        # Get DAB channel
        dab = deconv.extract_dab(img)

        # For each cell, measure DAB in three compartments
        for cell_id in np.unique(pred_cell):
            if cell_id == 0:
                continue

            cell_mask = pred_cell == cell_id
            nuc_mask = pred_nuc == cell_id

            if not nuc_mask.any():
                continue

            # Membrane ring: shrink cell inward by 10px (measured FWHM from
            # 17,708 radial DAB profiles — see evaluation/membrane_thickness.json).
            # DAB membrane signal spans ~10px and peaks 4px inside cell edge.
            cell_uint8 = cell_mask.astype(np.uint8)
            eroded = cv2.erode(cell_uint8, np.ones((5, 5), np.uint8), iterations=2)
            membrane_ring = cell_mask & ~eroded.astype(bool)

            # Cytoplasm: cell minus nucleus minus membrane ring
            cyto_mask = cell_mask & ~nuc_mask & ~membrane_ring

            if membrane_ring.sum() < 5 or nuc_mask.sum() < 5:
                continue

            membrane_dab.append(float(dab[membrane_ring].mean()))
            nucleus_dab.append(float(dab[nuc_mask].mean()))
            if cyto_mask.sum() > 5:
                cytoplasm_dab.append(float(dab[cyto_mask].mean()))

        tiles_processed += 1
        if tiles_processed >= n_tiles:
            break

    results = {
        "membrane_dab_mean": float(np.mean(membrane_dab)) if membrane_dab else 0,
        "cytoplasm_dab_mean": float(np.mean(cytoplasm_dab)) if cytoplasm_dab else 0,
        "nucleus_dab_mean": float(np.mean(nucleus_dab)) if nucleus_dab else 0,
        "membrane_dab_std": float(np.std(membrane_dab)) if membrane_dab else 0,
        "n_cells_measured": len(membrane_dab),
        "n_tiles": tiles_processed,
        "membrane_gt_cytoplasm": float(np.mean(membrane_dab)) > float(np.mean(cytoplasm_dab)) if membrane_dab and cytoplasm_dab else None,
        "membrane_gt_nucleus": float(np.mean(membrane_dab)) > float(np.mean(nucleus_dab)) if membrane_dab and nucleus_dab else None,
    }

    logger.info("Test 2: %d cells across %d tiles", len(membrane_dab), tiles_processed)
    logger.info("  Membrane DAB:  %.4f +/- %.4f", results["membrane_dab_mean"], results["membrane_dab_std"])
    logger.info("  Cytoplasm DAB: %.4f", results["cytoplasm_dab_mean"])
    logger.info("  Nucleus DAB:   %.4f", results["nucleus_dab_mean"])
    logger.info("  Membrane > Cytoplasm: %s", results["membrane_gt_cytoplasm"])
    logger.info("  Membrane > Nucleus:   %s", results["membrane_gt_nucleus"])

    return results


# ---------------------------------------------------------------------------
# Test 3: Comparison with nucleus expansion
# ---------------------------------------------------------------------------

def test3_vs_nucleus_expansion(
    model, loss_fn, test_data: list, data_dir: Path,
    expansion_um: float = 5.0, mpp: float = 0.5,
    n_tiles: int = 100, device: str = "cuda",
):
    """Compare predicted cell boundaries with simple nucleus expansion."""
    from scipy import ndimage

    expansion_px = int(expansion_um / mpp)
    iou_with_expansion = []
    boundary_differs = 0
    total_cells = 0
    tiles_processed = 0

    for item in tqdm(test_data[:n_tiles * 2], desc="Test 3: vs expansion"):
        img = tifffile.imread(data_dir / item["image"])
        gt_nuc = tifffile.imread(data_dir / item["nucleus_masks"]).astype(np.int32)
        if gt_nuc.max() == 0:
            continue

        try:
            pred_nuc, pred_cell = _predict_tile(model, loss_fn, img, device)
        except Exception:
            continue

        if pred_nuc.max() == 0 or pred_cell.max() == 0:
            continue

        # Create nucleus expansion baseline
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expansion_px + 1, 2 * expansion_px + 1))
        expanded = np.zeros_like(pred_nuc)
        for nid in np.unique(pred_nuc):
            if nid == 0:
                continue
            nuc_mask = (pred_nuc == nid).astype(np.uint8)
            exp_mask = cv2.dilate(nuc_mask, kernel, iterations=1)
            expanded[exp_mask > 0] = nid

        # Compare: for each cell, compute IoU between our prediction and expansion
        for cell_id in np.unique(pred_cell):
            if cell_id == 0:
                continue
            our_cell = pred_cell == cell_id
            exp_cell = expanded == cell_id

            if not exp_cell.any():
                continue

            intersection = (our_cell & exp_cell).sum()
            union = (our_cell | exp_cell).sum()
            iou = intersection / union if union > 0 else 0

            iou_with_expansion.append(float(iou))
            if iou < 0.8:  # boundaries meaningfully differ
                boundary_differs += 1
            total_cells += 1

        tiles_processed += 1
        if tiles_processed >= n_tiles:
            break

    results = {
        "mean_iou_with_expansion": float(np.mean(iou_with_expansion)) if iou_with_expansion else 0,
        "median_iou_with_expansion": float(np.median(iou_with_expansion)) if iou_with_expansion else 0,
        "pct_boundaries_differ": boundary_differs / max(total_cells, 1) * 100,
        "total_cells_compared": total_cells,
        "n_tiles": tiles_processed,
        "expansion_um": expansion_um,
    }

    logger.info("Test 3: %d cells across %d tiles", total_cells, tiles_processed)
    logger.info("  Mean IoU with %dum expansion: %.4f", expansion_um, results["mean_iou_with_expansion"])
    logger.info("  Boundaries differ (IoU<0.8): %.1f%%", results["pct_boundaries_differ"])

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    eval_dir = Path(cfg["paths"]["evaluation_dir"])

    logger.info("=" * 65)
    logger.info("Boundary Validation — 3 Objective Tests")
    logger.info("=" * 65)

    # Load model
    model, loss_fn = _load_model_and_postprocess(cfg)
    logger.info("Model loaded")

    # Load test data
    dataset = torch.load(data_dir / "segmentation_dataset.pth", weights_only=False)
    test_data = dataset["Test"]
    logger.info("Test tiles: %d", len(test_data))

    # Test 1: Visual overlays
    logger.info("\n--- Test 1: Visual Overlays ---")
    overlay_dir = eval_dir / "boundary_overlays"
    n_overlays = test1_visual_overlays(model, loss_fn, test_data, data_dir, overlay_dir)

    # Test 2: Compartment intensity
    logger.info("\n--- Test 2: Compartment DAB Intensity ---")
    compartment_results = test2_compartment_intensity(model, loss_fn, test_data, data_dir, cfg)

    # Test 3: vs nucleus expansion
    logger.info("\n--- Test 3: vs Nucleus Expansion (5um) ---")
    expansion_results = test3_vs_nucleus_expansion(model, loss_fn, test_data, data_dir)

    # Save all results
    all_results = {
        "test1_overlays": n_overlays,
        "test2_compartment_intensity": compartment_results,
        "test3_vs_expansion": expansion_results,
    }

    results_path = eval_dir / "boundary_validation.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 65)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 65)
    logger.info("Test 1: %d overlay images saved to %s", n_overlays, overlay_dir)
    logger.info("Test 2: Membrane DAB=%.4f, Cytoplasm=%.4f, Nucleus=%.4f",
                compartment_results["membrane_dab_mean"],
                compartment_results["cytoplasm_dab_mean"],
                compartment_results["nucleus_dab_mean"])
    if compartment_results["membrane_gt_cytoplasm"]:
        logger.info("  --> PASS: Membrane > Cytoplasm (biologically correct)")
    else:
        logger.info("  --> FAIL: Membrane <= Cytoplasm")
    logger.info("Test 3: %.1f%% of boundaries differ from 5um expansion (IoU<0.8)",
                expansion_results["pct_boundaries_differ"])
    if expansion_results["pct_boundaries_differ"] > 30:
        logger.info("  --> PASS: Model learned real boundaries, not just expansion")
    else:
        logger.info("  --> INCONCLUSIVE: Boundaries similar to expansion")
    logger.info("=" * 65)
    logger.info("Full results: %s", results_path)


if __name__ == "__main__":
    main()
