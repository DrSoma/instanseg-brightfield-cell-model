"""Detect DAB membrane pixels using BiomedParse or a SegFormer fallback.

BiomedParse (Microsoft Research) is a foundation model for biomedical image
segmentation that supports text-prompted zero-shot inference.  When available
it can segment membrane structures directly from a natural-language prompt
("cell membrane", "DAB membrane staining", etc.) with no training at all.

When BiomedParse is unavailable this script falls back to a pretrained
SegFormer-B2 from HuggingFace, fine-tuning only its decoder on our
self-supervised DAB ground truth -- exactly the same label strategy used by
``train_membrane_detector.py``.

After training (or zero-shot inference), the predicted membrane maps are
combined with existing nucleus masks via marker-controlled watershed to
produce improved cell instance segmentations.

Usage:
    # Zero-shot with BiomedParse (no training):
    python scripts/train_membrane_biomedparse.py --mode zeroshot

    # Zero-shot with custom prompt:
    python scripts/train_membrane_biomedparse.py --mode zeroshot \\
        --prompt "brown membrane ring"

    # Fine-tune (BiomedParse or SegFormer fallback):
    python scripts/train_membrane_biomedparse.py --mode finetune --epochs 20

    # Skip training, just generate masks from existing checkpoint:
    python scripts/train_membrane_biomedparse.py --skip-training
"""

from __future__ import annotations

import argparse
import importlib
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
# Backend availability probes
# ---------------------------------------------------------------------------


def _biomedparse_available() -> bool:
    """Return True if the biomedparse package can be imported."""
    try:
        importlib.import_module("biomedparse")
        return True
    except Exception:
        return False


def _transformers_available() -> bool:
    """Return True if HuggingFace transformers is importable."""
    try:
        importlib.import_module("transformers")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Dice loss
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation (operates on logits)."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: Predicted logits (B, 1, H, W).
            targets: Binary ground truth (B, 1, H, W) in {0, 1}.

        Returns:
            Scalar Dice loss.
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """BCEWithLogitsLoss + DiceLoss weighted sum."""

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


# ---------------------------------------------------------------------------
# BiomedParse wrapper
# ---------------------------------------------------------------------------


class BiomedParseMembraneDetector(nn.Module):
    """Wrapper around BiomedParse for membrane segmentation.

    In **zero-shot** mode the model is used purely for inference with a text
    prompt -- no parameters are updated.

    In **fine-tune** mode the vision encoder is frozen and only the
    decoder / prompt-encoder layers are trained.

    Args:
        device: Torch device.
        freeze_encoder: Whether to freeze the vision encoder.
    """

    def __init__(self, device: torch.device, freeze_encoder: bool = True) -> None:
        super().__init__()
        import biomedparse  # type: ignore[import-untyped]

        logger.info("Loading BiomedParse model ...")
        self.model = biomedparse.load_model()
        self.model.to(device)
        self._device = device

        if freeze_encoder:
            self._freeze_encoder()

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "BiomedParse loaded: %s total params, %s trainable",
            f"{total:,}",
            f"{trainable:,}",
        )

    def _freeze_encoder(self) -> None:
        """Freeze the vision encoder, keep decoder/prompt-encoder trainable."""
        frozen = 0
        for name, param in self.model.named_parameters():
            if "encoder" in name.lower() and "prompt" not in name.lower():
                param.requires_grad = False
                frozen += 1
        logger.info("Froze %d encoder parameter tensors", frozen)

    def predict_zeroshot(
        self,
        rgb: np.ndarray,
        prompt: str = "cell membrane",
    ) -> np.ndarray:
        """Run zero-shot text-prompted segmentation on a single RGB image.

        Args:
            rgb: RGB uint8 image (H, W, 3).
            prompt: Natural-language description of the target structure.

        Returns:
            Probability map (H, W) float32 in [0, 1].
        """
        import biomedparse  # type: ignore[import-untyped]

        self.model.eval()
        with torch.no_grad():
            result = biomedparse.predict(self.model, rgb, prompt)

        # BiomedParse returns either a dict with 'mask' or a numpy array
        if isinstance(result, dict):
            mask = result.get("mask", result.get("prediction", None))
            if mask is None:
                mask = next(iter(result.values()))
        else:
            mask = result

        mask = np.asarray(mask, dtype=np.float32)
        # Normalise to [0, 1] if not already
        if mask.max() > 1.0:
            mask = mask / 255.0
        # Collapse to 2-D if needed
        if mask.ndim == 3:
            mask = mask.mean(axis=-1) if mask.shape[-1] > 1 else mask[..., 0]
        return np.clip(mask, 0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for fine-tuning.

        Args:
            x: (B, 3, H, W) float32 in [0, 1].

        Returns:
            Logit tensor (B, 1, H, W).
        """
        out = self.model(x)
        if isinstance(out, dict):
            out = out.get("logits", out.get("pred_masks", next(iter(out.values()))))
        if out.ndim == 3:
            out = out.unsqueeze(1)
        if out.shape[1] != 1:
            out = out[:, :1]
        return out


# ---------------------------------------------------------------------------
# SegFormer fallback wrapper
# ---------------------------------------------------------------------------


class SegFormerMembraneDetector(nn.Module):
    """SegFormer-B2 fine-tuned for binary membrane segmentation.

    The pretrained model is loaded from HuggingFace
    (``nvidia/segformer-b2-finetuned-ade-512-512``) and its classification
    head is replaced with a single-channel output.  The encoder is optionally
    frozen so that only the decoder is trained.

    Args:
        device: Torch device.
        freeze_encoder: Whether to freeze the MixTransformer backbone.
    """

    def __init__(self, device: torch.device, freeze_encoder: bool = True) -> None:
        super().__init__()
        from transformers import SegformerForSemanticSegmentation  # type: ignore[import-untyped]

        logger.info("Loading SegFormer-B2 from HuggingFace (fallback) ...")
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
        )

        # Replace classification head: ADE has 150 classes -> we need 1
        in_channels = self.segformer.decode_head.classifier.in_channels
        self.segformer.decode_head.classifier = nn.Conv2d(
            in_channels, 1, kernel_size=1
        )
        self.segformer.config.num_labels = 1

        self.segformer.to(device)
        self._device = device

        if freeze_encoder:
            self._freeze_encoder()

        total = sum(p.numel() for p in self.segformer.parameters())
        trainable = sum(
            p.numel() for p in self.segformer.parameters() if p.requires_grad
        )
        logger.info(
            "SegFormer-B2 loaded: %s total params, %s trainable",
            f"{total:,}",
            f"{trainable:,}",
        )

    def _freeze_encoder(self) -> None:
        """Freeze the MixTransformer encoder."""
        frozen = 0
        for name, param in self.segformer.named_parameters():
            if name.startswith("segformer.encoder"):
                param.requires_grad = False
                frozen += 1
        logger.info("Froze %d encoder parameter tensors", frozen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, H, W) float32 in [0, 1].

        Returns:
            Logit tensor (B, 1, H, W) at the input spatial resolution.
        """
        out = self.segformer(pixel_values=x)
        logits = out.logits  # (B, 1, H/4, W/4)
        # Up-sample to input resolution
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MembraneDataset(Dataset):
    """RGB tiles with self-supervised DAB membrane labels.

    Identical label strategy to ``train_membrane_detector.py``: the DAB
    channel from stain deconvolution is thresholded to produce a binary
    membrane target.

    Args:
        tile_paths: Paths to RGB PNG tiles.
        deconvolver: Pre-initialised StainDeconvolver.
        dab_threshold: DAB concentration threshold for positive label.
        tile_size: Spatial size to resize tiles to (both H and W).
    """

    def __init__(
        self,
        tile_paths: list[Path],
        deconvolver: StainDeconvolver,
        dab_threshold: float = 0.10,
        tile_size: int = 512,
    ) -> None:
        self.tile_paths = tile_paths
        self.deconvolver = deconvolver
        self.dab_threshold = dab_threshold
        self.tile_size = tile_size

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one tile and its self-supervised label.

        Returns:
            (input, target): float32 tensors.
              - input: (3, H, W) in [0, 1].
              - target: (1, H, W) binary {0.0, 1.0}.
        """
        tile_path = self.tile_paths[idx]

        bgr = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read tile: {tile_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Resize if needed (SegFormer expects consistent spatial dims)
        h, w = rgb.shape[:2]
        if h != self.tile_size or w != self.tile_size:
            rgb = cv2.resize(
                rgb, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR
            )

        dab = self.deconvolver.extract_dab(rgb)
        membrane = (dab > self.dab_threshold).astype(np.float32)

        rgb_float = rgb.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(rgb_float).permute(2, 0, 1)
        target_tensor = torch.from_numpy(membrane).unsqueeze(0)

        return input_tensor, target_tensor


# ---------------------------------------------------------------------------
# Data helpers
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
    """Estimate neg/pos class weight for BCEWithLogitsLoss.

    Samples a subset of tiles, computes the membrane pixel fraction, and
    returns ``neg_count / pos_count`` so the loss balances both classes.

    Args:
        tile_paths: All tile paths.
        deconvolver: Stain deconvolver instance.
        dab_threshold: DAB threshold for positive label.
        max_sample: Maximum tiles to sample.
        seed: Random seed.

    Returns:
        Positive weight (>= 1.0).
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
        "Estimated pos_weight: %.2f (%.1f%% membrane across %d tiles)",
        weight,
        100.0 * total_pos / (total_pos + total_neg),
        len(sampled),
    )
    return weight


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_model(
    device: torch.device,
    freeze_encoder: bool = True,
) -> nn.Module:
    """Build the best available segmentation backbone.

    Attempts BiomedParse first; falls back to SegFormer-B2.

    Args:
        device: Target torch device.
        freeze_encoder: Freeze the vision encoder for fine-tuning.

    Returns:
        An ``nn.Module`` with a ``forward(x) -> (B, 1, H, W)`` interface.

    Raises:
        RuntimeError: If neither backend is available.
    """
    if _biomedparse_available():
        logger.info("BiomedParse detected -- using as segmentation backbone")
        return BiomedParseMembraneDetector(device, freeze_encoder=freeze_encoder)

    if _transformers_available():
        logger.info(
            "BiomedParse not found; falling back to SegFormer-B2 "
            "(nvidia/segformer-b2-finetuned-ade-512-512)"
        )
        return SegFormerMembraneDetector(device, freeze_encoder=freeze_encoder)

    raise RuntimeError(
        "Neither biomedparse nor transformers is installed. "
        "Install one of:\n"
        "  pip install biomedparse\n"
        "  pip install transformers torch\n"
    )


# ---------------------------------------------------------------------------
# Zero-shot inference
# ---------------------------------------------------------------------------


def run_zeroshot(
    model: BiomedParseMembraneDetector,
    tiles_dir: Path,
    output_dir: Path,
    prompt: str,
    membrane_threshold: float,
    cfg: dict,
    deconvolver: StainDeconvolver,
    masks_dir: Path,
) -> None:
    """Run zero-shot membrane prediction on all tiles.

    No training is performed.  BiomedParse predicts membrane probability
    from the text prompt, the output is thresholded, and watershed produces
    cell instance masks.

    Args:
        model: BiomedParseMembraneDetector in eval mode.
        tiles_dir: Root tiles directory.
        output_dir: Where to write cell mask TIFFs.
        prompt: Text prompt for BiomedParse.
        membrane_threshold: Probability threshold for binary membrane.
        cfg: Pipeline configuration.
        deconvolver: Stain deconvolver.
        masks_dir: Existing nucleus masks directory.
    """
    ws_cfg = cfg["watershed"]
    mpp: float = cfg["tile_extraction"].get("target_mpp", 0.5)
    max_cell_radius_px = ws_cfg["max_cell_radius_um"] / mpp

    slide_tiles = _collect_tiles_by_slide(tiles_dir)
    if not slide_tiles:
        logger.error("No tiles found under %s", tiles_dir)
        return

    total_tiles = 0
    total_cells = 0

    for slide_name, tile_paths in slide_tiles.items():
        slide_mask_dir = masks_dir / slide_name
        if not slide_mask_dir.is_dir():
            logger.info("Skipping slide %s: no existing nucleus masks", slide_name)
            continue

        slide_output_dir = output_dir / slide_name
        slide_output_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        for tp in tile_paths:
            stem = tp.stem
            nuc_mask_path = slide_mask_dir / f"{stem}_nuclei.tiff"
            if not nuc_mask_path.exists():
                continue

            bgr = cv2.imread(str(tp), cv2.IMREAD_COLOR)
            if bgr is None:
                logger.warning("Could not read tile %s; skipping", tp)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            prob_map = model.predict_zeroshot(rgb, prompt=prompt)
            membrane_binary = (prob_map > membrane_threshold).astype(np.uint8) * 255

            nucleus_labels = tifffile.imread(str(nuc_mask_path)).astype(np.int32)
            if nucleus_labels.max() == 0:
                cell_path = slide_output_dir / f"{stem}_cells.tiff"
                tifffile.imwrite(
                    str(cell_path),
                    np.zeros_like(nucleus_labels, dtype=np.uint16),
                    compression="zlib",
                )
                processed += 1
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

            cell_path = slide_output_dir / f"{stem}_cells.tiff"
            tifffile.imwrite(
                str(cell_path),
                cell_labels.astype(np.uint16),
                compression="zlib",
            )

            total_cells += int(cell_labels.max())
            processed += 1

        total_tiles += processed
        logger.info("Slide %s: %d tiles processed (zero-shot)", slide_name, processed)

    logger.info(
        "Zero-shot complete: %d tiles, %d cell instances", total_tiles, total_cells
    )


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


def finetune_model(
    model: nn.Module,
    tile_paths: list[Path],
    deconvolver: StainDeconvolver,
    save_path: Path,
    dab_threshold: float = 0.10,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    patience: int = 5,
    val_fraction: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Fine-tune the segmentation model on DAB-derived membrane labels.

    The vision encoder is already frozen (done in the model constructor);
    only the decoder / prompt-encoder parameters are updated.

    Args:
        model: BiomedParseMembraneDetector or SegFormerMembraneDetector.
        tile_paths: All tile paths.
        deconvolver: Stain deconvolver.
        save_path: Checkpoint save path.
        dab_threshold: DAB threshold for ground truth.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        patience: Early stopping patience.
        val_fraction: Validation split fraction.
        num_workers: DataLoader workers.
        seed: Random seed.
        device: Torch device (inferred from model if None).

    Returns:
        The model with best checkpoint loaded.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = next(model.parameters()).device

    # Dataset
    full_dataset = MembraneDataset(tile_paths, deconvolver, dab_threshold)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    logger.info("Dataset split: %d train, %d val (%d total)", n_train, n_val, n_total)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # Loss
    pos_weight_val = _estimate_pos_weight(
        tile_paths, deconvolver, dab_threshold, seed=seed
    )
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = CombinedLoss(pos_weight=pos_weight, bce_weight=0.5, dice_weight=0.5)

    # Optimiser -- only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.error("No trainable parameters -- check encoder freeze logic")
        sys.exit(1)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # AMP
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state: Optional[dict] = None

    logger.info("Starting fine-tuning: %d epochs, patience=%d", epochs, patience)

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs)
                # Ensure spatial dims match target
                if logits.shape[2:] != targets.shape[2:]:
                    logits = F.interpolate(
                        logits,
                        size=targets.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # -- Validate --
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(inputs)
                    if logits.shape[2:] != targets.shape[2:]:
                        logits = F.interpolate(
                            logits,
                            size=targets.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    loss = criterion(logits, targets)

                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %3d/%d  train=%.5f  val=%.5f  lr=%.2e",
            epoch, epochs, avg_train_loss, avg_val_loss, current_lr,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  -> New best val_loss=%.5f, saving checkpoint", best_val_loss)
        else:
            epochs_no_improve += 1
            logger.info(
                "  -> No improvement for %d/%d epochs (best=%.5f)",
                epochs_no_improve, patience, best_val_loss,
            )
            if epochs_no_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d exhausted)",
                    epoch, patience,
                )
                break

    # Save best
    if best_state is not None:
        model.load_state_dict(best_state)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path)
        logger.info("Best model saved to %s (val_loss=%.5f)", save_path, best_val_loss)
    else:
        logger.warning("No checkpoint saved (training may have been too short)")

    return model


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------


def generate_masks(
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
    """Generate cell masks by running the fine-tuned model + watershed.

    For each tile with existing nucleus masks, the model predicts membrane
    probability, which is thresholded and fed to watershed.

    Args:
        model: Fine-tuned segmentation model.
        cfg: Pipeline configuration.
        deconvolver: Stain deconvolver.
        tiles_dir: Root tiles directory.
        masks_dir: Existing nucleus masks directory.
        output_dir: Output directory for cell masks.
        device: Torch device.
        membrane_threshold: Probability threshold for binary membrane.
        batch_size: Inference batch size.
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

        # Collect tiles that have nucleus masks
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

        for batch_start in range(0, len(valid_tiles), batch_size):
            batch = valid_tiles[batch_start : batch_start + batch_size]

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

            input_batch = torch.stack(batch_tensors).to(device, non_blocking=True)

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_batch)
                if logits.shape[2:] != input_batch.shape[2:]:
                    logits = F.interpolate(
                        logits,
                        size=input_batch.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
            probs = torch.sigmoid(logits).cpu().numpy()

            rgb_idx = 0
            for i, (tp, nuc_mask_path) in enumerate(batch):
                if rgb_idx >= len(batch_rgbs):
                    break

                rgb = batch_rgbs[rgb_idx]
                rgb_idx += 1

                prob_map = probs[i, 0]
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
            "Slide %s complete: %d tiles processed", slide_name, len(valid_tiles)
        )

    logger.info(
        "Mask generation complete: %d tiles, %d total cell instances",
        total_tiles, total_cells,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse CLI arguments, run zero-shot or fine-tune, generate masks."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect DAB membrane pixels using BiomedParse (zero-shot or "
            "fine-tuned) with a SegFormer-B2 fallback, then generate improved "
            "cell masks via watershed."
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
        default=20,
        help="Training epochs for fine-tune mode (default: 20).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zeroshot", "finetune"],
        default="finetune",
        help="Operation mode (default: finetune).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="cell membrane",
        help="Text prompt for zero-shot mode (default: 'cell membrane').",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        default=False,
        help=(
            "Skip training and load an existing checkpoint. "
            "Useful for regenerating masks with an already-trained model."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training and inference (default: 8).",
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
    output_dir = data_dir / "masks_biomedparse"
    model_path = model_dir / "membrane_biomedparse.pth"

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

    # Device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info("Device: %s", device)

    # Collect tiles
    tile_paths = _collect_all_tile_paths(tiles_dir)
    if not tile_paths:
        logger.error("No tiles found under %s", tiles_dir)
        sys.exit(1)
    logger.info("Found %d tiles across all slides", len(tile_paths))

    # ------------------------------------------------------------------
    # Zero-shot mode
    # ------------------------------------------------------------------
    if args.mode == "zeroshot":
        if not _biomedparse_available():
            logger.error(
                "Zero-shot mode requires BiomedParse. "
                "Install it with: pip install biomedparse"
            )
            sys.exit(1)

        logger.info("=" * 72)
        logger.info("BiomedParse Zero-Shot Membrane Detection")
        logger.info("Prompt: %r", args.prompt)
        logger.info("=" * 72)

        model = BiomedParseMembraneDetector(device, freeze_encoder=True)
        model.eval()

        run_zeroshot(
            model=model,
            tiles_dir=tiles_dir,
            output_dir=output_dir,
            prompt=args.prompt,
            membrane_threshold=0.5,
            cfg=cfg,
            deconvolver=deconvolver,
            masks_dir=masks_dir,
        )

        logger.info("=" * 72)
        logger.info("Zero-shot masks saved to: %s", output_dir)
        logger.info("=" * 72)
        return

    # ------------------------------------------------------------------
    # Fine-tune mode
    # ------------------------------------------------------------------
    if args.skip_training:
        if not model_path.exists():
            logger.error(
                "Cannot skip training: checkpoint not found at %s", model_path
            )
            sys.exit(1)

        logger.info("Skipping training; loading checkpoint from %s", model_path)
        model = _build_model(device, freeze_encoder=True)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        logger.info("=" * 72)
        logger.info("Membrane Detection: Fine-Tune Mode")
        logger.info("=" * 72)

        backend = "BiomedParse" if _biomedparse_available() else "SegFormer-B2"
        logger.info("Backend         : %s", backend)
        logger.info("Epochs          : %d", args.epochs)
        logger.info("Batch size      : %d", args.batch_size)
        logger.info("Learning rate   : 1e-4")
        logger.info("DAB threshold   : %.2f", dab_threshold)
        logger.info("Early stopping  : patience=5")
        logger.info("Loss            : BCEWithLogits + Dice (0.5 / 0.5)")
        logger.info("Tiles available : %d", len(tile_paths))
        logger.info("=" * 72)

        model = _build_model(device, freeze_encoder=True)
        model = finetune_model(
            model=model,
            tile_paths=tile_paths,
            deconvolver=deconvolver,
            save_path=model_path,
            dab_threshold=dab_threshold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-4,
            patience=5,
            val_fraction=0.15,
            num_workers=num_workers,
            seed=seed,
            device=device,
        )

    # ------------------------------------------------------------------
    # Generate masks
    # ------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("Generating cell masks with fine-tuned membrane detector")
    logger.info("=" * 72)

    generate_masks(
        model=model,
        cfg=cfg,
        deconvolver=deconvolver,
        tiles_dir=tiles_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        device=device,
        membrane_threshold=0.5,
        batch_size=args.batch_size,
    )

    logger.info("=" * 72)
    logger.info("All done. Model: %s", model_path)
    logger.info("Masks: %s", output_dir)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
