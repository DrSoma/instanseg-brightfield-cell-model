"""Train a SegFormer-B2 model to detect DAB membrane pixels from RGB tiles.

An alternative to the vanilla U-Net membrane detector
(``scripts/train_membrane_detector.py``), this script leverages SegFormer's
transformer encoder for stronger global context --- the self-attention layers
can see the full membrane ring at once rather than relying on a small
receptive field.

The model is initialised from HuggingFace's ADE20K-pretrained SegFormer-B2
(~27M params) and fine-tuned with a two-phase schedule:

  Phase 1 (first ``freeze_epochs``): encoder frozen, only the decoder head is
  trained at a higher learning rate (``lr``).

  Phase 2 (remaining epochs): all layers unfrozen, trained end-to-end at a
  lower learning rate (``lr_unfreeze``).

If the ``transformers`` library is not installed, the script falls back to a
larger vanilla U-Net with channel widths [64, 128, 256, 512].

Training data is self-supervised: the DAB channel extracted via stain
deconvolution is thresholded to produce binary membrane labels, so no manual
annotation is needed.

After training, the learned detector is used to produce improved cell instance
masks via marker-controlled watershed (nucleus seeds + predicted membrane
boundaries).

Usage:
    python scripts/train_membrane_segformer.py
    python scripts/train_membrane_segformer.py --epochs 40 --batch-size 4
    python scripts/train_membrane_segformer.py --skip-training  # masks only
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
# Backend detection: prefer HuggingFace SegFormer, fall back to large U-Net
# ---------------------------------------------------------------------------

_HAS_TRANSFORMERS = False
try:
    from transformers import SegformerForSemanticSegmentation  # noqa: F401

    _HAS_TRANSFORMERS = True
except ImportError:
    logger.info(
        "transformers library not installed; "
        "falling back to enlarged U-Net (channels [64, 128, 256, 512])"
    )


# ---------------------------------------------------------------------------
# Fallback: larger U-Net (used when transformers is unavailable)
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
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class FallbackUNet(nn.Module):
    """Enlarged vanilla U-Net used when the transformers library is absent.

    Channel widths [64, 128, 256, 512] yield ~31M parameters, closer to
    SegFormer-B2's capacity than the baseline [32, 64, 128, 256] U-Net.

    Args:
        channels: Channel counts for the four encoder stages.
    """

    def __init__(self, channels: tuple[int, ...] = (64, 128, 256, 512)) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels

        self.inc = _DoubleConv(3, c0)
        self.down1 = _Down(c0, c1)
        self.down2 = _Down(c1, c2)
        self.down3 = _Down(c2, c3)
        self.bottleneck = _Down(c3, c3 * 2)
        self.up1 = _Up(c3 * 2, c3)
        self.up2 = _Up(c3, c2)
        self.up3 = _Up(c2, c1)
        self.up4 = _Up(c1, c0)
        self.outc = nn.Conv2d(c0, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# SegFormer wrapper
# ---------------------------------------------------------------------------


class MembraneSegFormer(nn.Module):
    """Thin wrapper around HuggingFace SegFormer-B2 for binary membrane segmentation.

    SegFormer outputs logits at 1/4 input resolution.  The wrapper
    bilinearly upsamples them to the original spatial size so that loss
    computation and downstream inference operate at full resolution.

    Args:
        pretrained_name: HuggingFace model identifier for the pretrained
            SegFormer checkpoint.
    """

    def __init__(
        self,
        pretrained_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
    ) -> None:
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            num_labels=1,  # binary: membrane vs not
            ignore_mismatched_sizes=True,
        )

    # -- Freeze / unfreeze helpers -----------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze the SegFormer encoder (Mix Transformer backbone)."""
        for param in self.model.segformer.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze the SegFormer encoder for end-to-end fine-tuning."""
        for param in self.model.segformer.parameters():
            param.requires_grad = True

    # -- Forward -----------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run SegFormer and upsample logits to input resolution.

        Args:
            pixel_values: (B, 3, H, W) float tensor in [0, 1].

        Returns:
            Logit tensor of shape (B, 1, H, W) at the input spatial size.
        """
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  # (B, 1, H/4, W/4)
        logits = F.interpolate(
            logits,
            size=pixel_values.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits  # (B, 1, H, W)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# SegFormer-B2 expects 512x512 input.
_SEGFORMER_SIZE = 512


class MembraneSegFormerDataset(Dataset):
    """PyTorch dataset that loads RGB tiles, resizes to 512x512, and
    generates DAB-based binary membrane labels.

    For each tile:
      - Input:  RGB image resized to 512x512, normalised to [0, 1], (3, H, W)
      - Target: binary membrane mask resized to 512x512, (1, H, W)

    Args:
        tile_paths: List of paths to RGB PNG tile images.
        deconvolver: Pre-initialised StainDeconvolver instance.
        dab_threshold: Threshold on DAB concentration for membrane label.
        target_size: Spatial size to resize tiles to.
    """

    def __init__(
        self,
        tile_paths: list[Path],
        deconvolver: StainDeconvolver,
        dab_threshold: float = 0.10,
        target_size: int = _SEGFORMER_SIZE,
    ) -> None:
        self.tile_paths = tile_paths
        self.deconvolver = deconvolver
        self.dab_threshold = dab_threshold
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile_path = self.tile_paths[idx]

        bgr = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read tile: {tile_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Stain deconvolution -> DAB channel (on original resolution)
        dab = self.deconvolver.extract_dab(rgb)
        membrane = (dab > self.dab_threshold).astype(np.float32)

        # Resize to target size
        rgb_resized = cv2.resize(
            rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR,
        )
        membrane_resized = cv2.resize(
            membrane, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST,
        )

        # Convert to tensors
        rgb_float = rgb_resized.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(rgb_float).permute(2, 0, 1)       # (3, H, W)
        target_tensor = torch.from_numpy(membrane_resized).unsqueeze(0)    # (1, H, W)

        return input_tensor, target_tensor


# ---------------------------------------------------------------------------
# Data collection helpers
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
# Loss: BCEWithLogitsLoss + DiceLoss (combined)
# ---------------------------------------------------------------------------


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy (with logits) and Dice loss.

    Using both losses produces better calibrated probabilities (from BCE) and
    strong overlap optimisation (from Dice), which is especially helpful for
    the imbalanced membrane-vs-background task.

    Args:
        bce_weight: Weight for the BCE component.
        dice_weight: Weight for the Dice component.
        smooth: Smoothing constant for Dice to avoid division by zero.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined BCE + Dice loss.

        Args:
            logits: Raw model output (B, 1, H, W).
            targets: Binary targets (B, 1, H, W) with values 0.0 or 1.0.

        Returns:
            Scalar loss value.
        """
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# Pos-weight estimation (for logging; not directly used by BCEDiceLoss)
# ---------------------------------------------------------------------------


def _estimate_membrane_fraction(
    tile_paths: list[Path],
    deconvolver: StainDeconvolver,
    dab_threshold: float,
    max_sample: int = 200,
    seed: int = 42,
) -> float:
    """Estimate the fraction of positive (membrane) pixels across a tile sample.

    Args:
        tile_paths: All available tile paths.
        deconvolver: Stain deconvolver instance.
        dab_threshold: DAB threshold for membrane label.
        max_sample: Maximum number of tiles to sample.
        seed: Random seed.

    Returns:
        Membrane pixel fraction in [0, 1].
    """
    rng = random.Random(seed)
    sampled = rng.sample(tile_paths, min(max_sample, len(tile_paths)))

    total_pos = 0
    total_pixels = 0

    for tp in sampled:
        bgr = cv2.imread(str(tp), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dab = deconvolver.extract_dab(rgb)
        membrane = dab > dab_threshold
        total_pos += int(membrane.sum())
        total_pixels += int(membrane.size)

    fraction = total_pos / max(total_pixels, 1)
    logger.info(
        "Membrane pixel fraction: %.2f%% across %d sampled tiles",
        fraction * 100.0,
        len(sampled),
    )
    return fraction


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_model(device: torch.device) -> nn.Module:
    """Build the membrane segmentation model.

    Returns a ``MembraneSegFormer`` if the transformers library is available,
    otherwise falls back to ``FallbackUNet`` with channels [64, 128, 256, 512].

    Args:
        device: Target device.

    Returns:
        The model placed on ``device``.
    """
    if _HAS_TRANSFORMERS:
        logger.info("Building SegFormer-B2 (pretrained on ADE20K)")
        model = MembraneSegFormer(
            pretrained_name="nvidia/segformer-b2-finetuned-ade-512-512",
        )
    else:
        logger.info("Building fallback U-Net (channels [64, 128, 256, 512])")
        model = FallbackUNet(channels=(64, 128, 256, 512))

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters:     %s", f"{total_params:,}")
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_membrane_segformer(
    tile_paths: list[Path],
    deconvolver: StainDeconvolver,
    save_path: Path,
    dab_threshold: float = 0.10,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    lr_unfreeze: float = 1e-5,
    freeze_epochs: int = 5,
    patience: int = 7,
    val_fraction: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
    device_str: str = "cuda",
    weight_decay: float = 0.01,
) -> nn.Module:
    """Train the SegFormer (or fallback U-Net) and save the best checkpoint.

    Two-phase training schedule:
      - Phase 1 (epochs 1..freeze_epochs): encoder frozen, decoder-only at ``lr``.
      - Phase 2 (epochs freeze_epochs+1..epochs): all layers, AdamW at ``lr_unfreeze``.

    If the fallback U-Net is used (no transformers), all layers are trained
    from the start with learning rate ``lr``.

    Args:
        tile_paths: All available tile paths.
        deconvolver: Stain deconvolver instance.
        save_path: Where to save the best model weights.
        dab_threshold: DAB threshold for membrane ground truth.
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for phase 1 (decoder only) or fallback U-Net.
        lr_unfreeze: Learning rate for phase 2 (full model).
        freeze_epochs: Number of epochs to keep the encoder frozen.
        patience: Early stopping patience (epochs without val improvement).
        val_fraction: Fraction of data for validation.
        num_workers: DataLoader worker processes.
        seed: Random seed for reproducibility.
        device_str: Device string ("cuda" or "cpu").
        weight_decay: AdamW weight decay.

    Returns:
        The trained model (loaded with best weights).
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
    full_dataset = MembraneSegFormerDataset(tile_paths, deconvolver, dab_threshold)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    logger.info("Dataset split: %d train, %d val (%d total)", n_train, n_val, n_total)

    # Estimate membrane fraction for logging
    _estimate_membrane_fraction(tile_paths, deconvolver, dab_threshold, seed=seed)

    # Auto-reduce batch size on OOM
    effective_batch_size = batch_size

    def _make_loaders(bs: int) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )
        return train_loader, val_loader

    train_loader, val_loader = _make_loaders(effective_batch_size)

    # Build model
    model = _build_model(device)

    is_segformer = isinstance(model, MembraneSegFormer)

    # Phase 1: freeze encoder if SegFormer
    if is_segformer:
        model.freeze_encoder()
        logger.info(
            "Phase 1: encoder frozen for first %d epochs (decoder-only training)",
            freeze_epochs,
        )

    # Loss
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

    # Optimizer (will be rebuilt when entering phase 2)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # AMP scaler
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training state
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state: Optional[dict] = None
    phase = 1 if (is_segformer and freeze_epochs > 0) else 2

    logger.info(
        "Starting training: %d epochs, patience=%d, batch_size=%d",
        epochs, patience, effective_batch_size,
    )

    for epoch in range(1, epochs + 1):
        # --- Phase transition (SegFormer only) ---
        if is_segformer and phase == 1 and epoch > freeze_epochs:
            logger.info("=" * 60)
            logger.info(
                "Phase 2: unfreezing encoder at epoch %d, lr -> %.1e",
                epoch, lr_unfreeze,
            )
            logger.info("=" * 60)
            model.unfreeze_encoder()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr_unfreeze,
                weight_decay=weight_decay,
            )
            # Reset the scaler state for the new optimiser
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            phase = 2
            trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logger.info("Trainable parameters after unfreeze: %s", f"{trainable:,}")

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(inputs)
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            except torch.cuda.OutOfMemoryError:
                # Auto-reduce batch size on OOM
                torch.cuda.empty_cache()
                effective_batch_size = max(1, effective_batch_size // 2)
                logger.warning(
                    "OOM detected; reducing batch size to %d and rebuilding loaders",
                    effective_batch_size,
                )
                train_loader, val_loader = _make_loaders(effective_batch_size)
                break  # restart this epoch with new loaders

            train_loss_sum += loss.item()
            train_batches += 1

        if train_batches == 0:
            # OOM on the very first batch after reduction; skip this epoch
            continue

        avg_train_loss = train_loss_sum / train_batches

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
        phase_tag = f"[phase {phase}]" if is_segformer else ""
        logger.info(
            "Epoch %3d/%d %s  train_loss=%.5f  val_loss=%.5f",
            epoch, epochs, phase_tag, avg_train_loss, avg_val_loss,
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
                    "Early stopping at epoch %d (patience=%d exhausted)",
                    epoch, patience,
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
    model: nn.Module,
    cfg: dict,
    deconvolver: StainDeconvolver,
    tiles_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    device: torch.device,
    membrane_threshold: float = 0.5,
    batch_size: int = 8,
) -> None:
    """Generate improved cell masks using the trained membrane detector.

    For each tile that has existing nucleus masks, run the model to predict
    membrane probability, threshold it, and use watershed to grow cell
    instances from nucleus seeds.

    Args:
        model: Trained MembraneSegFormer or FallbackUNet.
        cfg: Pipeline configuration dictionary.
        deconvolver: Stain deconvolver instance.
        tiles_dir: Root tiles directory (data/tiles).
        masks_dir: Existing masks directory (data/masks).
        output_dir: Output directory for improved masks.
        device: Torch device.
        membrane_threshold: Threshold on predicted probability for binary membrane.
        batch_size: Batch size for inference.
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
            stem = tp.stem
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

            batch_tensors: list[torch.Tensor] = []
            batch_rgbs: list[np.ndarray] = []
            batch_valid: list[int] = []  # track which batch entries loaded OK

            for i, (tp, _) in enumerate(batch):
                bgr = cv2.imread(str(tp), cv2.IMREAD_COLOR)
                if bgr is None:
                    logger.warning("Could not read tile %s; skipping", tp)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                batch_rgbs.append(rgb)
                batch_valid.append(i)

                # Resize to 512x512 for model inference
                rgb_resized = cv2.resize(
                    rgb, (_SEGFORMER_SIZE, _SEGFORMER_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )
                rgb_float = rgb_resized.astype(np.float32) / 255.0
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
            for bi, orig_i in enumerate(batch_valid):
                tp, nuc_mask_path = batch[orig_i]
                rgb = batch_rgbs[bi]
                orig_h, orig_w = rgb.shape[:2]

                # Upsample probability map back to original tile resolution
                prob_map = probs[bi, 0]  # (512, 512)
                if (orig_h, orig_w) != (_SEGFORMER_SIZE, _SEGFORMER_SIZE):
                    prob_map = cv2.resize(
                        prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR,
                    )

                membrane_binary = (prob_map > membrane_threshold).astype(np.uint8) * 255

                nucleus_labels = tifffile.imread(str(nuc_mask_path)).astype(np.int32)

                if nucleus_labels.max() == 0:
                    stem = tp.stem
                    cell_path = slide_output_dir / f"{stem}_cells.tiff"
                    tifffile.imwrite(
                        str(cell_path),
                        np.zeros_like(nucleus_labels, dtype=np.uint16),
                        compression="zlib",
                    )
                    continue

                dab = deconvolver.extract_dab(rgb)

                cell_labels = segment_cells(
                    nucleus_labels=nucleus_labels,
                    dab_channel=dab,
                    membrane_mask=membrane_binary,
                    max_cell_radius_px=max_cell_radius_px,
                    compactness=float(ws_cfg["compactness"]),
                    distance_sigma=float(ws_cfg["distance_sigma"]),
                )

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
            slide_name, len(valid_tiles),
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
            "Train a SegFormer-B2 membrane detector (or fallback large U-Net) "
            "on DAB-deconvolved tiles and generate improved cell masks."
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
        help="Override the number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size (default: 8).",
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
    output_dir = data_dir / "masks_segformer"
    model_path = model_dir / "membrane_segformer.pth"

    # Training parameters (config defaults with CLI overrides)
    epochs = args.epochs if args.epochs is not None else 30
    batch_size = args.batch_size if args.batch_size is not None else 8
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

    backend = "SegFormer-B2" if _HAS_TRANSFORMERS else "Fallback U-Net [64,128,256,512]"

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

        model = _build_model(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        logger.info("=" * 72)
        logger.info("Membrane SegFormer Training")
        logger.info("=" * 72)
        logger.info("Backend         : %s", backend)
        logger.info("Epochs          : %d", epochs)
        logger.info("Batch size      : %d", batch_size)
        logger.info("Freeze epochs   : %d", 5)
        logger.info("LR (phase 1)    : 1e-4")
        logger.info("LR (phase 2)    : 1e-5")
        logger.info("Weight decay    : 0.01")
        logger.info("DAB threshold   : %.2f", dab_threshold)
        logger.info("Loss            : BCEWithLogitsLoss + DiceLoss")
        logger.info("Early stopping  : patience=7")
        logger.info("Tiles available : %d", len(tile_paths))
        logger.info("=" * 72)

        model = train_membrane_segformer(
            tile_paths=tile_paths,
            deconvolver=deconvolver,
            save_path=model_path,
            dab_threshold=dab_threshold,
            epochs=epochs,
            batch_size=batch_size,
            lr=1e-4,
            lr_unfreeze=1e-5,
            freeze_epochs=5,
            patience=7,
            val_fraction=0.15,
            num_workers=num_workers,
            seed=seed,
            device_str=device_str,
            weight_decay=0.01,
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
    logger.info("All done. Backend: %s", backend)
    logger.info("Model:  %s", model_path)
    logger.info("Masks:  %s", output_dir)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
