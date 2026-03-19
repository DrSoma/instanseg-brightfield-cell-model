"""Train a binary U-Net to detect DAB membrane pixels from RGB tiles.

The training data is self-supervised: the DAB channel extracted via stain
deconvolution is thresholded to produce a binary membrane label, so no manual
annotation is needed.

After training, the learned membrane detector is used to produce improved cell
instance masks via marker-controlled watershed (nucleus seeds + predicted
membrane boundaries).

Usage:
    python scripts/train_membrane_detector.py
    python scripts/train_membrane_detector.py --epochs 100 --batch-size 8
    python scripts/train_membrane_detector.py --skip-training  # masks only
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver, build_stain_matrix
from instanseg_brightfield.watershed import segment_cells

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# U-Net architecture
# ---------------------------------------------------------------------------


class _DoubleConv(nn.Module):
    """Two consecutive (Conv2d -> BatchNorm -> ReLU) blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """Encoder block: MaxPool followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class _Up(nn.Module):
    """Decoder block: upsample, concatenate skip, then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims differ by 1 due to odd input sizes
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class MembraneUNet(nn.Module):
    """Minimal U-Net for binary membrane segmentation.

    Architecture: 4 encoder blocks, 4 decoder blocks, channel progression
    [32, 64, 128, 256]. Input is a 3-channel RGB tile normalised to [0, 1];
    output is a single-channel logit map (apply sigmoid for probabilities).

    Args:
        channels: Channel counts for the four encoder stages.
    """

    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256)) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels

        # Encoder
        self.inc = _DoubleConv(3, c0)
        self.down1 = _Down(c0, c1)
        self.down2 = _Down(c1, c2)
        self.down3 = _Down(c2, c3)

        # Bottleneck
        self.bottleneck = _Down(c3, c3 * 2)

        # Decoder
        self.up1 = _Up(c3 * 2, c3)
        self.up2 = _Up(c3, c2)
        self.up3 = _Up(c2, c1)
        self.up4 = _Up(c1, c0)

        # Output head (single channel logit)
        self.outc = nn.Conv2d(c0, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W) in [0, 1].

        Returns:
            Logit tensor of shape (B, 1, H, W).
        """
        x1 = self.inc(x)       # (B, c0, H, W)
        x2 = self.down1(x1)    # (B, c1, H/2, W/2)
        x3 = self.down2(x2)    # (B, c2, H/4, W/4)
        x4 = self.down3(x3)    # (B, c3, H/8, W/8)
        x5 = self.bottleneck(x4)  # (B, c3*2, H/16, W/16)

        x = self.up1(x5, x4)   # (B, c3, H/8, W/8)
        x = self.up2(x, x3)    # (B, c2, H/4, W/4)
        x = self.up3(x, x2)    # (B, c1, H/2, W/2)
        x = self.up4(x, x1)    # (B, c0, H, W)

        return self.outc(x)    # (B, 1, H, W)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MembraneDataset(Dataset):
    """PyTorch dataset that loads RGB tiles and generates DAB-based labels.

    For each tile:
      - Input: RGB image normalised to [0, 1], shape (3, H, W)
      - Target: binary membrane mask from DAB thresholding, shape (1, H, W)

    The DAB channel is extracted via stain deconvolution and thresholded at
    ``dab_threshold`` to produce the ground truth.

    Args:
        tile_paths: List of paths to RGB PNG tile images.
        deconvolver: Pre-initialised StainDeconvolver instance.
        dab_threshold: Threshold on DAB concentration for membrane label.
    """

    def __init__(
        self,
        tile_paths: list[Path],
        deconvolver: StainDeconvolver,
        dab_threshold: float = 0.10,
    ) -> None:
        self.tile_paths = tile_paths
        self.deconvolver = deconvolver
        self.dab_threshold = dab_threshold

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one tile and generate its membrane label.

        Returns:
            Tuple of (input, target) where:
              - input: float32 tensor (3, H, W) in [0, 1]
              - target: float32 tensor (1, H, W) with values 0.0 or 1.0
        """
        tile_path = self.tile_paths[idx]

        # Load RGB image
        bgr = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read tile: {tile_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Stain deconvolution -> DAB channel
        dab = self.deconvolver.extract_dab(rgb)

        # Threshold DAB to produce binary membrane label
        membrane = (dab > self.dab_threshold).astype(np.float32)

        # Convert to tensors
        rgb_float = rgb.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(rgb_float).permute(2, 0, 1)  # (3, H, W)
        target_tensor = torch.from_numpy(membrane).unsqueeze(0)       # (1, H, W)

        return input_tensor, target_tensor


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def _collect_all_tile_paths(tiles_dir: Path) -> list[Path]:
    """Collect all PNG tile paths from data/tiles/{slide}/*.png.

    Args:
        tiles_dir: Root tiles directory.

    Returns:
        Sorted list of tile paths across all slides.
    """
    tile_paths: list[Path] = []
    if not tiles_dir.is_dir():
        return tile_paths
    for slide_dir in sorted(tiles_dir.iterdir()):
        if not slide_dir.is_dir():
            continue
        tile_paths.extend(sorted(slide_dir.glob("*.png")))
    return tile_paths


def _collect_tiles_by_slide(tiles_dir: Path) -> dict[str, list[Path]]:
    """Group tile paths by slide name.

    Args:
        tiles_dir: Root tiles directory.

    Returns:
        Mapping from slide name to sorted list of tile paths.
    """
    slide_tiles: dict[str, list[Path]] = {}
    if not tiles_dir.is_dir():
        return slide_tiles
    for slide_dir in sorted(tiles_dir.iterdir()):
        if not slide_dir.is_dir():
            continue
        pngs = sorted(slide_dir.glob("*.png"))
        if pngs:
            slide_tiles[slide_dir.name] = pngs
    return slide_tiles


# ---------------------------------------------------------------------------
# Pos-weight estimation
# ---------------------------------------------------------------------------


def _estimate_pos_weight(
    tile_paths: list[Path],
    deconvolver: StainDeconvolver,
    dab_threshold: float,
    max_sample: int = 200,
    seed: int = 42,
) -> float:
    """Estimate the positive class weight for BCEWithLogitsLoss.

    Samples a subset of tiles, computes the fraction of membrane pixels, and
    returns neg_count / pos_count so that the loss treats both classes equally.

    Args:
        tile_paths: All available tile paths.
        deconvolver: Stain deconvolver instance.
        dab_threshold: DAB threshold for membrane label.
        max_sample: Maximum number of tiles to sample.
        seed: Random seed.

    Returns:
        Positive weight (float >= 1.0).
    """
    rng = random.Random(seed)
    sampled = rng.sample(tile_paths, min(max_sample, len(tile_paths)))

    total_pos = 0
    total_neg = 0

    for tp in sampled:
        bgr = cv2.imread(str(tp), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dab = deconvolver.extract_dab(rgb)
        membrane = dab > dab_threshold
        n_pos = int(membrane.sum())
        n_neg = int(membrane.size - n_pos)
        total_pos += n_pos
        total_neg += n_neg

    if total_pos == 0:
        logger.warning("No positive membrane pixels found; using pos_weight=1.0")
        return 1.0

    weight = total_neg / total_pos
    logger.info(
        "Estimated pos_weight: %.2f (%.1f%% membrane pixels across %d tiles)",
        weight,
        100.0 * total_pos / (total_pos + total_neg),
        len(sampled),
    )
    return weight


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_membrane_detector(
    tile_paths: list[Path],
    deconvolver: StainDeconvolver,
    save_path: Path,
    dab_threshold: float = 0.10,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 10,
    val_fraction: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
    device_str: str = "cuda",
) -> MembraneUNet:
    """Train the membrane U-Net and save the best checkpoint.

    Args:
        tile_paths: All available tile paths.
        deconvolver: Stain deconvolver instance.
        save_path: Where to save the best model weights.
        dab_threshold: DAB threshold for membrane ground truth.
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for Adam.
        patience: Early stopping patience (epochs without val improvement).
        val_fraction: Fraction of data for validation.
        num_workers: DataLoader worker processes.
        seed: Random seed for reproducibility.
        device_str: Device string ("cuda" or "cpu").

    Returns:
        The trained MembraneUNet (loaded with best weights).
    """
    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device setup
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info("Training device: %s", device)

    # Build dataset and split
    full_dataset = MembraneDataset(tile_paths, deconvolver, dab_threshold)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    logger.info("Dataset split: %d train, %d val (%d total)", n_train, n_val, n_total)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # Model
    model = MembraneUNet(channels=(32, 64, 128, 256)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("MembraneUNet parameters: %s", f"{total_params:,}")

    # Loss with class-imbalance handling
    pos_weight_val = _estimate_pos_weight(
        tile_paths, deconvolver, dab_threshold, seed=seed,
    )
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # AMP scaler
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training loop with early stopping
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state: Optional[dict] = None

    logger.info("Starting training: %d epochs, patience=%d", epochs, patience)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(inputs)
                    loss = criterion(logits, targets)

                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)

        # --- Logging and early stopping ---
        logger.info(
            "Epoch %3d/%d  train_loss=%.5f  val_loss=%.5f",
            epoch, epochs, avg_train_loss, avg_val_loss,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(
                "  -> New best val_loss=%.5f, saving checkpoint", best_val_loss,
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                "  -> No improvement for %d/%d epochs (best=%.5f)",
                epochs_without_improvement, patience, best_val_loss,
            )
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d exhausted)", epoch, patience,
                )
                break

    # Load best weights and save
    if best_state is not None:
        model.load_state_dict(best_state)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path)
        logger.info("Best model saved to %s (val_loss=%.5f)", save_path, best_val_loss)
    else:
        logger.warning("No checkpoint was saved (training may have been too short)")

    return model


# ---------------------------------------------------------------------------
# Improved mask generation
# ---------------------------------------------------------------------------


def generate_improved_masks(
    model: MembraneUNet,
    cfg: dict,
    deconvolver: StainDeconvolver,
    tiles_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    device: torch.device,
    membrane_threshold: float = 0.5,
    batch_size: int = 128,
) -> None:
    """Generate improved cell masks using the trained membrane detector.

    For each tile that has existing nucleus masks, run the membrane detector
    to predict membrane probability, threshold it, and use watershed to grow
    cell instances from nucleus seeds.

    Args:
        model: Trained MembraneUNet.
        cfg: Pipeline configuration dictionary.
        deconvolver: Stain deconvolver instance.
        tiles_dir: Root tiles directory (data/tiles).
        masks_dir: Existing masks directory (data/masks).
        output_dir: Output directory for improved masks.
        device: Torch device.
        membrane_threshold: Threshold on predicted probability for binary membrane.
        batch_size: Batch size for membrane detector inference.
    """
    ws_cfg = cfg["watershed"]
    mpp: float = cfg["tile_extraction"].get("target_mpp", 0.5)
    max_cell_radius_px = ws_cfg["max_cell_radius_um"] / mpp

    slide_tiles = _collect_tiles_by_slide(tiles_dir)
    if not slide_tiles:
        logger.error("No tiles found under %s", tiles_dir)
        return

    model.eval()
    use_amp = device.type == "cuda"

    total_tiles = 0
    total_cells = 0

    for slide_name, tile_paths in slide_tiles.items():
        slide_mask_dir = masks_dir / slide_name
        if not slide_mask_dir.is_dir():
            logger.info("Skipping slide %s: no existing masks", slide_name)
            continue

        slide_output_dir = output_dir / slide_name
        slide_output_dir.mkdir(parents=True, exist_ok=True)

        # Collect tiles that have existing nucleus masks
        valid_tiles: list[tuple[Path, Path]] = []
        for tp in tile_paths:
            stem = tp.stem  # e.g. "100672_48576"
            nuc_mask_path = slide_mask_dir / f"{stem}_nuclei.tiff"
            if nuc_mask_path.exists():
                valid_tiles.append((tp, nuc_mask_path))

        if not valid_tiles:
            logger.info("Skipping slide %s: no nucleus masks found", slide_name)
            continue

        logger.info(
            "Processing slide %s: %d tiles with nucleus masks",
            slide_name, len(valid_tiles),
        )

        # Process in batches for GPU efficiency
        for batch_start in range(0, len(valid_tiles), batch_size):
            batch = valid_tiles[batch_start:batch_start + batch_size]

            # Load tile images for membrane prediction
            batch_tensors: list[torch.Tensor] = []
            batch_rgbs: list[np.ndarray] = []

            for tp, _ in batch:
                bgr = cv2.imread(str(tp), cv2.IMREAD_COLOR)
                if bgr is None:
                    logger.warning("Could not read tile %s; skipping", tp)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                batch_rgbs.append(rgb)
                rgb_float = rgb.astype(np.float32) / 255.0
                t = torch.from_numpy(rgb_float).permute(2, 0, 1)
                batch_tensors.append(t)

            if not batch_tensors:
                continue

            # Run membrane detector
            input_batch = torch.stack(batch_tensors).to(device, non_blocking=True)

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_batch)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)

            # Process each tile in the batch
            rgb_idx = 0
            for i, (tp, nuc_mask_path) in enumerate(batch):
                if rgb_idx >= len(batch_rgbs):
                    break

                rgb = batch_rgbs[rgb_idx]
                rgb_idx += 1

                prob_map = probs[i, 0]  # (H, W)

                # Threshold to binary membrane
                membrane_binary = (prob_map > membrane_threshold).astype(np.uint8) * 255

                # Load existing nucleus labels
                nucleus_labels = tifffile.imread(str(nuc_mask_path)).astype(np.int32)

                if nucleus_labels.max() == 0:
                    # No nuclei: save empty cell mask
                    stem = tp.stem
                    cell_path = slide_output_dir / f"{stem}_cells.tiff"
                    tifffile.imwrite(
                        str(cell_path),
                        np.zeros_like(nucleus_labels, dtype=np.uint16),
                        compression="zlib",
                    )
                    continue

                # DAB channel for watershed energy
                dab = deconvolver.extract_dab(rgb)

                # Watershed with learned membrane boundaries
                cell_labels = segment_cells(
                    nucleus_labels=nucleus_labels,
                    dab_channel=dab,
                    membrane_mask=membrane_binary,
                    max_cell_radius_px=max_cell_radius_px,
                    compactness=float(ws_cfg["compactness"]),
                    distance_sigma=float(ws_cfg["distance_sigma"]),
                )

                # Save
                stem = tp.stem
                cell_path = slide_output_dir / f"{stem}_cells.tiff"
                tifffile.imwrite(
                    str(cell_path),
                    cell_labels.astype(np.uint16),
                    compression="zlib",
                )

                n_cells = int(cell_labels.max())
                total_cells += n_cells
                total_tiles += 1

        logger.info(
            "Slide %s complete: %d tiles processed",
            slide_name, sum(1 for _ in valid_tiles),
        )

    logger.info(
        "Improved mask generation complete: %d tiles, %d total cell instances",
        total_tiles, total_cells,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse arguments, train model, generate improved masks."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary U-Net membrane detector on DAB-deconvolved tiles "
            "and generate improved cell masks."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size (default: 16).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        default=False,
        help=(
            "Skip training and use an existing model checkpoint. "
            "Useful for regenerating masks with an already-trained detector."
        ),
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

    data_dir = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    tiles_dir = data_dir / "tiles"
    masks_dir = data_dir / "masks"
    output_dir = data_dir / "masks_membrane_detector"
    model_path = model_dir / "membrane_detector.pth"

    # Training parameters (config defaults with CLI overrides)
    epochs = args.epochs if args.epochs is not None else 50
    batch_size = args.batch_size if args.batch_size is not None else 128
    seed: int = cfg["training"].get("seed", 42)
    num_workers: int = cfg["training"].get("num_workers", 4)
    device_str: str = cfg["nucleus_detection"].get("device", "cuda")
    dab_threshold = 0.10

    # Stain deconvolver
    sc = cfg["stain_deconvolution"]
    stain_matrix = build_stain_matrix(
        hematoxylin=sc["hematoxylin"],
        dab=sc["dab"],
        residual=sc["residual"],
    )
    deconvolver = StainDeconvolver(stain_matrix)

    # Collect tiles
    tile_paths = _collect_all_tile_paths(tiles_dir)
    if not tile_paths:
        logger.error("No tiles found under %s", tiles_dir)
        sys.exit(1)
    logger.info("Found %d tiles across all slides", len(tile_paths))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    if args.skip_training:
        if not model_path.exists():
            logger.error(
                "Cannot skip training: model not found at %s", model_path,
            )
            sys.exit(1)
        logger.info("Skipping training; loading existing model from %s", model_path)

        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
        device = torch.device(device_str)

        model = MembraneUNet(channels=(32, 64, 128, 256)).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        logger.info("=" * 72)
        logger.info("Membrane Detector Training")
        logger.info("=" * 72)
        logger.info("Epochs          : %d", epochs)
        logger.info("Batch size      : %d", batch_size)
        logger.info("Learning rate   : 1e-3")
        logger.info("DAB threshold   : %.2f", dab_threshold)
        logger.info("Early stopping  : patience=10")
        logger.info("Tiles available : %d", len(tile_paths))
        logger.info("=" * 72)

        model = train_membrane_detector(
            tile_paths=tile_paths,
            deconvolver=deconvolver,
            save_path=model_path,
            dab_threshold=dab_threshold,
            epochs=epochs,
            batch_size=batch_size,
            lr=1e-3,
            patience=10,
            val_fraction=0.15,
            num_workers=num_workers,
            seed=seed,
            device_str=device_str,
        )
        device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Generate improved masks
    # ------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("Generating improved cell masks with trained membrane detector")
    logger.info("=" * 72)

    generate_improved_masks(
        model=model,
        cfg=cfg,
        deconvolver=deconvolver,
        tiles_dir=tiles_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        device=next(model.parameters()).device,
        membrane_threshold=0.5,
        batch_size=batch_size,
    )

    logger.info("=" * 72)
    logger.info("All done. Model: %s", model_path)
    logger.info("Improved masks: %s", output_dir)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
