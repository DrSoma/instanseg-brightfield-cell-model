"""Extract and remap weights from TorchScript brightfield_nuclei to a dual-head UNet.

Loads the TorchScript `brightfield_nuclei` model, strips the `fcn.` key prefix,
and maps the single-head weights to a dual-head (cells_and_nuclei) architecture.

The nuclei output heads are copied to initialize the cells output heads.
The result is a model_weights.pth checkpoint + experiment_log.csv that
InstanSeg's training code can load directly for fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from instanseg_brightfield.config import get_config_hash, load_config

logger = logging.getLogger(__name__)


def _extract_torchscript_state_dict(model_name: str = "brightfield_nuclei") -> dict[str, torch.Tensor]:
    """Load a TorchScript InstanSeg model and return its state_dict."""
    from instanseg import InstanSeg

    logger.info("Loading TorchScript model '%s' ...", model_name)
    model = InstanSeg(model_name, device="cpu")
    ts_model = model.instanseg
    sd = ts_model.state_dict()
    logger.info("Extracted %d parameters from TorchScript model", len(sd))
    return dict(sd)


def _remap_to_dual_head(
    ts_sd: dict[str, torch.Tensor],
    n_sigma: int = 2,
    dim_coords: int = 2,
    dim_seeds: int = 1,
) -> dict[str, torch.Tensor]:
    """Remap TorchScript state_dict to dual-head InstanSeg_UNet.

    Mapping:
    - fcn.encoder.*           → encoder.*           (shared encoder)
    - fcn.decoders.0.decoder.* → decoders.0.decoder.* (shared decoder body)
    - fcn.decoders.0.final_block.{0,1,2}.* → decoders.0.final_block.{0,1,2}.* (nuclei heads)
    - fcn.decoders.0.final_block.{0,1,2}.* → decoders.0.final_block.{3,4,5}.* (cells heads, COPIED)
    - fcn.pixel_classifier.*  → pixel_classifier.*  (MLP)

    Args:
        ts_sd: TorchScript state_dict.
        n_sigma: Number of sigma channels (determines final_block head count).
        dim_coords: Spatial embedding dimensions.
        dim_seeds: Seed map channels.

    Returns:
        Remapped state_dict for dual-head InstanSeg_UNet.
    """
    new_sd: dict[str, torch.Tensor] = {}
    n_heads_per_target = 3  # final_block has 3 sub-modules per target: coords, sigma, seeds

    for key, value in ts_sd.items():
        # Skip the duplicate top-level pixel_classifier (keep the fcn. one)
        if key.startswith("pixel_classifier."):
            continue

        if not key.startswith("fcn."):
            logger.warning("Unexpected key (no 'fcn.' prefix): %s — skipping", key)
            continue

        # Strip 'fcn.' prefix
        stripped = key[4:]  # "fcn.X" → "X"

        # Direct copy: encoder and decoder body
        new_sd[stripped] = value.clone()

        # For final_block heads: also create cells copies at offset indices
        if "final_block" in stripped:
            # Parse the head index: decoders.0.final_block.{idx}.{rest}
            parts = stripped.split(".")
            fb_idx = int(parts[3])  # 0, 1, or 2

            if fb_idx < n_heads_per_target:
                # Create the cells copy at fb_idx + n_heads_per_target
                cells_idx = fb_idx + n_heads_per_target
                cells_key = f"{parts[0]}.{parts[1]}.{parts[2]}.{cells_idx}.{'.'.join(parts[4:])}"
                new_sd[cells_key] = value.clone()

    logger.info(
        "Remapped %d → %d parameters (added %d cells head params)",
        len(ts_sd),
        len(new_sd),
        len(new_sd) - len([k for k in ts_sd if k.startswith("fcn.")]),
    )
    return new_sd


def _verify_extraction(
    ts_sd: dict[str, torch.Tensor],
    new_sd: dict[str, torch.Tensor],
) -> bool:
    """Verify that encoder weights match exactly between original and remapped."""
    mismatches = 0
    for key, value in ts_sd.items():
        if not key.startswith("fcn.encoder."):
            continue
        new_key = key[4:]  # strip 'fcn.'
        if new_key not in new_sd:
            logger.error("Missing encoder key in remapped: %s", new_key)
            mismatches += 1
            continue
        if not torch.equal(value, new_sd[new_key]):
            logger.error("Value mismatch for %s", new_key)
            mismatches += 1

    if mismatches == 0:
        logger.info("Verification PASSED: all encoder weights match exactly")
    else:
        logger.error("Verification FAILED: %d mismatches", mismatches)
    return mismatches == 0


def _build_and_load_model(
    new_sd: dict[str, torch.Tensor],
    cfg: dict,
) -> torch.nn.Module:
    """Build a dual-head InstanSeg_UNet and load the remapped weights.

    Returns the loaded model for sanity checking.
    """
    from instanseg.utils.model_loader import build_model_from_dict

    build_dict = {
        "model_str": "InstanSeg_UNet",
        "layers": tuple(cfg["training"]["layers"]),
        "dim_in": cfg["training"]["dim_in"],
        "n_sigma": cfg["training"]["n_sigma"],
        "dim_coords": cfg["training"]["dim_coords"],
        "dim_seeds": cfg["training"]["dim_seeds"],
        "norm": cfg["training"]["norm"],
        "cells_and_nuclei": True,
        "multihead": False,
        "dropprob": 0.0,
    }

    model = build_model_from_dict(build_dict)

    # Filter to only keys present in the model
    model_keys = set(model.state_dict().keys())
    loadable = {k: v for k, v in new_sd.items() if k in model_keys}
    extra = {k for k in new_sd if k not in model_keys}
    missing = model_keys - set(new_sd.keys())

    if extra:
        logger.info("Extra keys (pixel_classifier, etc.): %d — stored separately", len(extra))
    if missing:
        logger.warning("Missing keys in extracted weights: %s", missing)

    model.load_state_dict(loadable, strict=False)
    logger.info("Loaded %d / %d model parameters", len(loadable), len(model_keys))

    return model


def _save_checkpoint(
    new_sd: dict[str, torch.Tensor],
    cfg: dict,
    output_dir: Path,
    experiment_name: str,
) -> Path:
    """Save the remapped weights as an InstanSeg-compatible checkpoint.

    Creates:
    - {output_dir}/{experiment_name}/model_weights.pth
    - {output_dir}/{experiment_name}/experiment_log.csv
    """
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save model_weights.pth — must include a valid optimizer state_dict
    # because InstanSeg unconditionally calls optimizer.load_state_dict()
    from instanseg.utils.model_loader import build_model_from_dict
    dummy_model = build_model_from_dict({
        "model_str": "InstanSeg_UNet",
        "layers": tuple(cfg["training"]["layers"]),
        "dim_in": cfg["training"]["dim_in"],
        "n_sigma": cfg["training"]["n_sigma"],
        "dim_coords": cfg["training"]["dim_coords"],
        "dim_seeds": cfg["training"]["dim_seeds"],
        "norm": cfg["training"]["norm"],
        "cells_and_nuclei": True,
        "multihead": False,
        "dropprob": 0.0,
    })
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-4)

    checkpoint = {
        "model_state_dict": new_sd,
        "optimizer_state_dict": dummy_optimizer.state_dict(),
        "f1_score": 0.0,
        "epoch": 0,
    }
    weights_path = exp_dir / "model_weights.pth"
    torch.save(checkpoint, weights_path)

    # Compute hash for provenance
    with open(weights_path, "rb") as f:
        weights_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    logger.info("Saved weights to %s (sha256: %s)", weights_path, weights_hash)

    # Save experiment_log.csv (required by InstanSeg's load_model)
    tcfg = cfg["training"]
    log_data = {
        "model_str": "InstanSeg_UNet",
        "layers": str(tuple(tcfg["layers"])),
        "dim_in": tcfg["dim_in"],
        "dim_out": (tcfg["dim_coords"] + tcfg["n_sigma"] + tcfg["dim_seeds"]) * 2,
        "n_sigma": tcfg["n_sigma"],
        "dim_coords": tcfg["dim_coords"],
        "dim_seeds": tcfg["dim_seeds"],
        "norm": tcfg["norm"],
        "cells_and_nuclei": True,
        "target_segmentation": tcfg["target_segmentation"],
        "multihead": False,
        "dropprob": 0.0,
        "pixel_size": cfg["tile_extraction"]["target_mpp"],
        "channel_invariant": False,
        "feature_engineering": "0",
        "adaptor_net_str": "None",
        "mlp_width": tcfg["mlp_width"],
        "source": "extracted_from_brightfield_nuclei",
        "weights_hash": weights_hash,
    }

    csv_path = exp_dir / "experiment_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_data.keys()))
        writer.writeheader()
        writer.writerow(log_data)
    logger.info("Saved experiment_log.csv to %s", csv_path)

    return exp_dir


def _sanity_check(model: torch.nn.Module, cfg: dict) -> None:
    """Run a forward pass to verify the model produces the expected output shape."""
    tile_size = cfg["tile_extraction"]["tile_size"]
    dummy = torch.randn(1, 3, tile_size, tile_size)

    model.eval()
    with torch.inference_mode():
        output = model(dummy)

    tcfg = cfg["training"]
    expected_channels = (tcfg["dim_coords"] + tcfg["n_sigma"] + tcfg["dim_seeds"]) * 2

    assert output.shape[0] == 1, f"Batch dim mismatch: {output.shape}"
    assert output.shape[1] == expected_channels, (
        f"Channel mismatch: got {output.shape[1]}, expected {expected_channels}"
    )
    assert output.shape[2] == tile_size, f"Height mismatch: {output.shape}"
    assert output.shape[3] == tile_size, f"Width mismatch: {output.shape}"

    logger.info(
        "Sanity check PASSED: output shape %s (expected %d channels)",
        list(output.shape), expected_channels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract weights from TorchScript brightfield_nuclei for dual-head fine-tuning.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--source-model", type=str, default="brightfield_nuclei",
        help="Name of the InstanSeg model to extract from (default: brightfield_nuclei).",
    )
    parser.add_argument(
        "--output-name", type=str, default=None,
        help="Output experiment folder name (default: <experiment_name>_pretrained).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config(args.config)
    model_dir = Path(cfg["paths"]["model_dir"])
    experiment_name = args.output_name or f"{cfg['training']['experiment_name']}_pretrained"

    logger.info("=" * 72)
    logger.info("Weight Extraction: %s → dual-head %s", args.source_model, experiment_name)
    logger.info("=" * 72)

    # Step 1: Extract TorchScript state_dict
    ts_sd = _extract_torchscript_state_dict(args.source_model)

    # Step 2: Remap to dual-head architecture
    new_sd = _remap_to_dual_head(
        ts_sd,
        n_sigma=cfg["training"]["n_sigma"],
        dim_coords=cfg["training"]["dim_coords"],
        dim_seeds=cfg["training"]["dim_seeds"],
    )

    # Step 3: Verify encoder weights preserved
    if not _verify_extraction(ts_sd, new_sd):
        logger.error("Extraction verification failed — aborting")
        sys.exit(1)

    # Step 4: Build dual-head model and load weights
    model = _build_and_load_model(new_sd, cfg)

    # Step 5: Sanity check forward pass
    _sanity_check(model, cfg)

    # Step 6: Save checkpoint
    exp_dir = _save_checkpoint(new_sd, cfg, model_dir, experiment_name)

    logger.info("=" * 72)
    logger.info("Done! Pre-trained weights saved to: %s", exp_dir)
    logger.info("")
    logger.info("To fine-tune, run:")
    logger.info("  python scripts/04_train.py --pretrained-folder %s", exp_dir)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
