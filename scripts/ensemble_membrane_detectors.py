"""Ensemble membrane detectors for improved cell boundary ground truth.

Combines predictions from multiple membrane detection models (U-Net, MedSAM2,
BiomedParse) using various fusion strategies:
  - Average: mean probability across models
  - Majority vote: pixel is membrane if >=2 of 3 models agree
  - Intersection: pixel is membrane if ALL models agree (most conservative)
  - Best + DAB: single best model intersected with raw DAB threshold
  - Best + Fluorescence: consensus with fluorescence_nuclei_and_cells teacher

After fusion, runs watershed to generate improved cell instance masks.

Usage:
    python scripts/ensemble_membrane_detectors.py --strategy average
    python scripts/ensemble_membrane_detectors.py --strategy majority_vote
    python scripts/ensemble_membrane_detectors.py --strategy all  # try all strategies
"""

from __future__ import annotations

import argparse
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
from instanseg_brightfield.watershed import segment_cells

logger = logging.getLogger(__name__)

STRATEGIES = ["average", "majority_vote", "intersection", "best_plus_dab", "all"]


def _load_unet(model_path: Path, device: str = "cuda"):
    """Load the vanilla U-Net membrane detector."""
    if not model_path.exists():
        logger.warning("U-Net model not found at %s", model_path)
        return None
    from scripts.train_membrane_detector import MembraneUNet
    model = MembraneUNet(channels=(32, 64, 128, 256))
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model


def _predict_membrane_unet(model, tile_rgb: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Run U-Net inference, return probability map (H, W) in [0, 1]."""
    tensor = torch.from_numpy(tile_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
    return torch.sigmoid(logits[0, 0]).cpu().numpy()


def _predict_membrane_dab(tile_rgb: np.ndarray, deconv: StainDeconvolver, threshold: float = 0.10) -> np.ndarray:
    """Raw DAB threshold as membrane probability (0 or 1)."""
    dab = deconv.extract_dab(tile_rgb)
    return (dab > threshold).astype(np.float32)


def _fuse_predictions(predictions: list[np.ndarray], strategy: str) -> np.ndarray:
    """Fuse membrane probability maps from multiple models.

    Args:
        predictions: List of (H, W) float32 arrays in [0, 1].
        strategy: Fusion strategy name.

    Returns:
        Binary membrane mask (H, W), uint8, 255=membrane.
    """
    if not predictions:
        return np.zeros((512, 512), dtype=np.uint8)

    stack = np.stack(predictions, axis=0)  # (N, H, W)

    if strategy == "average":
        fused = stack.mean(axis=0)
        binary = (fused > 0.5).astype(np.uint8) * 255

    elif strategy == "majority_vote":
        votes = (stack > 0.5).astype(np.int32).sum(axis=0)
        threshold = len(predictions) / 2
        binary = (votes > threshold).astype(np.uint8) * 255

    elif strategy == "intersection":
        all_agree = np.all(stack > 0.5, axis=0)
        binary = all_agree.astype(np.uint8) * 255

    elif strategy == "best_plus_dab":
        # Use first prediction (assumed best model) intersected with last (assumed DAB)
        best = stack[0] > 0.5
        dab = stack[-1] > 0.5
        binary = (best & dab).astype(np.uint8) * 255

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return binary


def generate_ensemble_masks(
    cfg: dict,
    strategy: str = "average",
    device: str = "cuda",
):
    """Generate cell masks using ensembled membrane detectors."""
    data_dir = Path(cfg["paths"]["data_dir"])
    sc = cfg["stain_deconvolution"]
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))
    ws_cfg = cfg["watershed"]
    mpp = cfg["tile_extraction"]["target_mpp"]
    max_cell_radius_px = ws_cfg["max_cell_radius_um"] / mpp

    # Load available models
    models = {}

    unet = _load_unet(Path("models/membrane_detector.pth"), device)
    if unet:
        models["unet"] = unet
        logger.info("Loaded U-Net membrane detector")

    # MedSAM2 and BiomedParse would be loaded here if available
    medsam_path = Path("models/membrane_medsam2.pth")
    if medsam_path.exists():
        logger.info("MedSAM2 model found — loading")
        # models["medsam2"] = _load_medsam2(medsam_path, device)
    else:
        logger.info("MedSAM2 model not found — skipping")

    biomedparse_path = Path("models/membrane_biomedparse.pth")
    if biomedparse_path.exists():
        logger.info("BiomedParse model found — loading")
        # models["biomedparse"] = _load_biomedparse(biomedparse_path, device)
    else:
        logger.info("BiomedParse model not found — skipping")

    if not models:
        logger.error("No membrane detector models available!")
        return

    logger.info("Ensemble strategy: %s with %d models: %s", strategy, len(models), list(models.keys()))

    # Process tiles
    out_dir = data_dir / f"masks_ensemble_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = torch.load(data_dir / "segmentation_dataset_base.pth", weights_only=False)
    all_items = []
    for split in ["Train", "Validation", "Test"]:
        all_items.extend(base_dataset[split])

    processed = 0
    for item in tqdm(all_items, desc=f"Ensemble ({strategy})"):
        img = tifffile.imread(data_dir / item["image"])
        nuc_path = data_dir / item["nucleus_masks"]
        if not nuc_path.exists():
            continue
        nucleus_labels = tifffile.imread(nuc_path).astype(np.int32)
        if nucleus_labels.max() == 0:
            continue

        # Collect predictions from each model
        predictions = []
        for model_name, model in models.items():
            if model_name == "unet":
                pred = _predict_membrane_unet(model, img, device)
            else:
                pred = np.zeros_like(img[:, :, 0], dtype=np.float32)
            predictions.append(pred)

        # Always include raw DAB as last prediction
        predictions.append(_predict_membrane_dab(img, deconv))

        # Fuse
        membrane = _fuse_predictions(predictions, strategy)

        # Watershed
        dab = deconv.extract_dab(img)
        cell_labels = segment_cells(
            nucleus_labels, dab, membrane,
            max_cell_radius_px=max_cell_radius_px,
            compactness=float(ws_cfg["compactness"]),
            distance_sigma=float(ws_cfg["distance_sigma"]),
        )

        # Save
        slide = Path(item["image"]).parts[1]
        tile_stem = Path(item["image"]).stem
        slide_dir = out_dir / slide
        slide_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(slide_dir / f"{tile_stem}_cells.tiff"),
                         cell_labels.astype(np.uint16), compression="zlib")
        processed += 1

    logger.info("Ensemble (%s): processed %d tiles, masks in %s", strategy, processed, out_dir)
    return out_dir


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Ensemble membrane detectors for cell boundary generation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--strategy", type=str, default="average", choices=STRATEGIES)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.strategy == "all":
        for strategy in ["average", "majority_vote", "intersection", "best_plus_dab"]:
            logger.info("\n" + "=" * 60)
            logger.info("Running strategy: %s", strategy)
            logger.info("=" * 60)
            generate_ensemble_masks(cfg, strategy=strategy, device=args.device)
    else:
        generate_ensemble_masks(cfg, strategy=args.strategy, device=args.device)


if __name__ == "__main__":
    main()
