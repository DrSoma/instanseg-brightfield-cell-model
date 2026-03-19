"""Train MedSAM 2 (or SAM / ViT-UNet fallback) to detect DAB membrane pixels.

Uses a segmentation foundation model -- preferring MedSAM 2 (SAM fine-tuned on
medical images), falling back to vanilla SAM, and finally to a ViT-based U-Net
when neither foundation model is available.

The approach mirrors ``train_membrane_detector.py``: self-supervised labels are
derived from DAB stain deconvolution, so no manual annotation is required.
After training, the learned detector drives marker-controlled watershed to
produce cell instance masks from nucleus seeds.

Usage:
    python scripts/train_membrane_medsam2.py
    python scripts/train_membrane_medsam2.py --mode finetune --epochs 20
    python scripts/train_membrane_medsam2.py --mode zeroshot --skip-training
    python scripts/train_membrane_medsam2.py --batch-size 2 --epochs 30
"""

from __future__ import annotations

import argparse
import importlib
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

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
# Model backend discovery
# ---------------------------------------------------------------------------

_BACKEND_MEDSAM2 = "medsam2"
_BACKEND_SAM = "sam"
_BACKEND_VIT_UNET = "vit_unet"


def _try_import(module_name: str) -> bool:
    """Return True if *module_name* is importable."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _pip_install(package: str) -> bool:
    """Attempt to install *package* via pip; return True on success."""
    logger.info("Attempting to install '%s' via pip...", package)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        logger.warning("pip install '%s' failed", package)
        return False


def resolve_backend(preferred: str | None = None) -> str:
    """Determine which segmentation backend to use.

    Probes for MedSAM 2, vanilla SAM, and finally falls back to a lightweight
    ViT-based U-Net that needs no external model weights.

    Args:
        preferred: Explicitly request ``"medsam2"``, ``"sam"``, or
            ``"vit_unet"``.  If *None*, auto-detect in priority order.

    Returns:
        One of ``"medsam2"``, ``"sam"``, or ``"vit_unet"``.
    """
    if preferred == _BACKEND_VIT_UNET:
        logger.info("Backend forced to ViT-UNet (user request)")
        return _BACKEND_VIT_UNET

    # --- MedSAM 2 -----------------------------------------------------------
    if preferred in (None, _BACKEND_MEDSAM2):
        if _try_import("medsam"):
            logger.info("MedSAM 2 package detected")
            return _BACKEND_MEDSAM2
        if preferred is None and _pip_install("medsam"):
            if _try_import("medsam"):
                logger.info("MedSAM 2 installed successfully")
                return _BACKEND_MEDSAM2

    # --- Vanilla SAM ---------------------------------------------------------
    if preferred in (None, _BACKEND_SAM):
        if _try_import("segment_anything"):
            logger.info("segment-anything package detected")
            return _BACKEND_SAM
        if preferred is None and _pip_install("segment-anything"):
            if _try_import("segment_anything"):
                logger.info("segment-anything installed successfully")
                return _BACKEND_SAM

    # --- ViT-UNet fallback ---------------------------------------------------
    logger.warning(
        "Neither MedSAM 2 nor segment-anything available; "
        "falling back to built-in ViT-UNet"
    )
    return _BACKEND_VIT_UNET


# ---------------------------------------------------------------------------
# ViT-UNet fallback (no external dependencies beyond torch)
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    """Split an image into non-overlapping patches and project to an embedding."""

    def __init__(
        self, img_size: int = 256, patch_size: int = 16, in_ch: int = 3, embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, N, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class _TransformerBlock(nn.Module):
    """Standard pre-norm transformer encoder block."""

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class _DecoderBlock(nn.Module):
    """Upsample + Conv decoder block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class ViTUNet(nn.Module):
    """Lightweight ViT encoder + convolutional decoder for binary segmentation.

    Used as a pure-PyTorch fallback when neither MedSAM 2 nor vanilla SAM is
    available.  The architecture is deliberately compact (12 M parameters)
    so it trains fast with limited data.

    Args:
        img_size: Expected spatial resolution (square).
        patch_size: Patch size for the ViT stem.
        embed_dim: Transformer hidden dimension.
        depth: Number of transformer blocks.
        n_heads: Number of attention heads.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 6,
        n_heads: int = 6,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # tokens per side

        self.patch_embed = _PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.grid_size ** 2, embed_dim) * 0.02
        )
        self.blocks = nn.Sequential(*[
            _TransformerBlock(embed_dim, n_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder: progressively upsample from (grid_size, grid_size) to (img_size, img_size)
        ch = embed_dim
        decoder_layers: list[nn.Module] = []
        n_ups = 0
        s = self.grid_size
        while s < img_size:
            out_ch = max(ch // 2, 32)
            decoder_layers.append(_DecoderBlock(ch, out_ch))
            ch = out_ch
            s *= 2
            n_ups += 1
        self.decoder = nn.Sequential(*decoder_layers)
        self.head = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, H, W) in [0, 1].

        Returns:
            Logit map (B, 1, H, W).
        """
        B = x.shape[0]
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, D)
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        # Reshape tokens to spatial grid
        g = self.grid_size
        feat = tokens.transpose(1, 2).reshape(B, -1, g, g)  # (B, D, g, g)

        feat = self.decoder(feat)
        logits = self.head(feat)

        # Ensure output matches input spatial size exactly
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)

        return logits


# ---------------------------------------------------------------------------
# SAM / MedSAM adapter -- wraps the mask decoder for fine-tuning
# ---------------------------------------------------------------------------


class SAMMembraneAdapter(nn.Module):
    """Wraps a SAM / MedSAM model for membrane detection fine-tuning.

    The image encoder is frozen; only the mask decoder and prompt encoder
    are trained.  A lightweight convolutional head maps the multi-mask
    output to a single-channel membrane logit map.

    Args:
        sam_model: A loaded SAM or MedSAM model object.
        freeze_encoder: Whether to freeze the image encoder weights.
        img_size: Target image size for SAM (typically 1024).
    """

    def __init__(
        self,
        sam_model: Any,
        freeze_encoder: bool = True,
        img_size: int = 1024,
    ) -> None:
        super().__init__()
        self.sam = sam_model
        self.img_size = img_size

        if freeze_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
            logger.info("SAM image encoder frozen (%d parameters held fixed)",
                        sum(p.numel() for p in self.sam.image_encoder.parameters()))

        trainable = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
        logger.info("SAM trainable parameters: %s", f"{trainable:,}")

        # Head that collapses multi-mask output to single-channel logit map
        # SAM mask decoder outputs (B, num_masks, H_low, W_low)
        # We project to (B, 1, H_low, W_low) then upsample
        self.merge_head = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def _get_image_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Run the frozen image encoder.

        Args:
            x: (B, 3, img_size, img_size) in [0, 1].

        Returns:
            Image embedding tensor from SAM encoder.
        """
        # SAM expects images preprocessed with its own transform; we pass
        # normalised [0,1] -> [0,255] range as SAM's preprocessing expects.
        with torch.no_grad():
            return self.sam.image_encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing membrane logit map.

        Generates dense point prompts from the image and runs through the
        mask decoder.

        Args:
            x: (B, 3, H, W) normalised to [0, 1].

        Returns:
            Logit map (B, 1, H, W).
        """
        B, _, H, W = x.shape

        # Resize to SAM's expected input size if necessary
        if H != self.img_size or W != self.img_size:
            x_resized = F.interpolate(
                x, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )
        else:
            x_resized = x

        image_embeddings = self._get_image_embedding(x_resized)

        # Use no-prompt mode: generate sparse embeddings from prompt encoder
        # with no actual points (SAM will produce full-image masks)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None,
        )
        # Expand embeddings to match batch size
        sparse_embeddings = sparse_embeddings.expand(B, -1, -1)
        dense_embeddings = dense_embeddings.expand(B, -1, -1, -1)

        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        # low_res_masks: (B, num_masks, H_low, W_low) -- typically (B, 4, 256, 256)

        # Merge multiple mask channels to single logit
        merged = self.merge_head(low_res_masks)  # (B, 1, H_low, W_low)

        # Upsample to original resolution
        if merged.shape[2:] != (H, W):
            merged = F.interpolate(merged, size=(H, W), mode="bilinear", align_corners=False)

        return merged


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _load_medsam2_model(device: torch.device) -> Any:
    """Load the MedSAM 2 model.

    Returns:
        A MedSAM model instance.

    Raises:
        RuntimeError: If model cannot be loaded.
    """
    try:
        from medsam import medsam_model  # type: ignore[import-untyped]
        model = medsam_model.build_medsam()
        model = model.to(device)
        logger.info("MedSAM 2 model loaded successfully")
        return model
    except Exception:
        # Try alternative import paths common in different medsam versions
        try:
            import medsam  # type: ignore[import-untyped]
            if hasattr(medsam, "build_model"):
                model = medsam.build_model()
                model = model.to(device)
                logger.info("MedSAM 2 model loaded via medsam.build_model()")
                return model
        except Exception:
            pass
        raise RuntimeError("Failed to load MedSAM 2 model from installed package")


def _load_sam_model(device: torch.device) -> Any:
    """Load a vanilla SAM model (ViT-B by default).

    Attempts to load ViT-B checkpoint from standard locations; if no checkpoint
    is found, builds the model without pretrained weights.

    Returns:
        A SAM model instance.

    Raises:
        RuntimeError: If SAM cannot be loaded at all.
    """
    from segment_anything import sam_model_registry  # type: ignore[import-untyped]

    model_type = "vit_b"
    checkpoint_candidates = [
        Path("models/sam_vit_b_01ec64.pth"),
        Path.home() / ".cache" / "sam" / "sam_vit_b_01ec64.pth",
        Path("sam_vit_b_01ec64.pth"),
    ]

    checkpoint: Optional[Path] = None
    for cp in checkpoint_candidates:
        if cp.exists():
            checkpoint = cp
            break

    if checkpoint is not None:
        logger.info("Loading SAM %s from checkpoint: %s", model_type, checkpoint)
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    else:
        logger.warning(
            "No SAM checkpoint found; building %s without pretrained weights. "
            "Download sam_vit_b_01ec64.pth and place it in models/ for best results.",
            model_type,
        )
        sam = sam_model_registry[model_type]()

    sam = sam.to(device)
    logger.info("SAM model (%s) loaded successfully", model_type)
    return sam


def build_model(
    backend: str,
    device: torch.device,
    img_size: int = 256,
    freeze_encoder: bool = True,
) -> nn.Module:
    """Build the membrane detection model for the given backend.

    Args:
        backend: One of ``"medsam2"``, ``"sam"``, or ``"vit_unet"``.
        device: Torch device.
        img_size: Tile size (used for ViT-UNet patch calculation).
        freeze_encoder: Freeze SAM/MedSAM image encoder during fine-tuning.

    Returns:
        An ``nn.Module`` producing (B, 1, H, W) logit output.
    """
    if backend == _BACKEND_MEDSAM2:
        try:
            sam_model = _load_medsam2_model(device)
            return SAMMembraneAdapter(sam_model, freeze_encoder=freeze_encoder, img_size=1024).to(device)
        except RuntimeError as exc:
            logger.warning("MedSAM 2 load failed (%s); trying SAM fallback", exc)
            backend = _BACKEND_SAM

    if backend == _BACKEND_SAM:
        try:
            sam_model = _load_sam_model(device)
            return SAMMembraneAdapter(sam_model, freeze_encoder=freeze_encoder, img_size=1024).to(device)
        except Exception as exc:
            logger.warning("SAM load failed (%s); falling back to ViT-UNet", exc)
            backend = _BACKEND_VIT_UNET

    # ViT-UNet fallback
    model = ViTUNet(img_size=img_size, patch_size=16, embed_dim=384, depth=6, n_heads=6)
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    logger.info("ViT-UNet model built: %s parameters", f"{total:,}")
    return model


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """Differentiable soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """BCE + Dice combined loss.

    Args:
        bce_weight: Weight for the BCEWithLogitsLoss component.
        dice_weight: Weight for the DiceLoss component.
        pos_weight: Positive class weight for BCE (class imbalance).
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MedSAMMembraneDataset(Dataset):
    """Dataset that loads RGB tiles and generates DAB-threshold labels.

    For SAM-based models the tiles are resized to ``target_size`` during
    ``__getitem__`` so the dataloader can produce uniform batches.

    Args:
        tile_paths: Paths to RGB PNG tiles.
        deconvolver: StainDeconvolver instance.
        dab_threshold: DAB concentration threshold for membrane labels.
        target_size: Resize tiles to this square resolution (0 = no resize).
        augment: Apply basic data augmentation (horizontal/vertical flips).
    """

    def __init__(
        self,
        tile_paths: list[Path],
        deconvolver: StainDeconvolver,
        dab_threshold: float = 0.10,
        target_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.tile_paths = tile_paths
        self.deconvolver = deconvolver
        self.dab_threshold = dab_threshold
        self.target_size = target_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile_path = self.tile_paths[idx]

        bgr = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read tile: {tile_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # DAB -> binary membrane label
        dab = self.deconvolver.extract_dab(rgb)
        membrane = (dab > self.dab_threshold).astype(np.float32)

        # Resize if requested
        if self.target_size > 0 and (
            rgb.shape[0] != self.target_size or rgb.shape[1] != self.target_size
        ):
            rgb = cv2.resize(rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            membrane = cv2.resize(membrane, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                rgb = np.flip(rgb, axis=1).copy()
                membrane = np.flip(membrane, axis=1).copy()
            if random.random() > 0.5:
                rgb = np.flip(rgb, axis=0).copy()
                membrane = np.flip(membrane, axis=0).copy()

        # To tensors
        rgb_float = rgb.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(rgb_float).permute(2, 0, 1)   # (3, H, W)
        target_tensor = torch.from_numpy(membrane).unsqueeze(0)        # (1, H, W)

        return input_tensor, target_tensor


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _collect_all_tile_paths(tiles_dir: Path) -> list[Path]:
    """Collect all PNG tile paths from data/tiles/{slide}/*.png."""
    tile_paths: list[Path] = []
    if not tiles_dir.is_dir():
        return tile_paths
    for slide_dir in sorted(tiles_dir.iterdir()):
        if not slide_dir.is_dir():
            continue
        tile_paths.extend(sorted(slide_dir.glob("*.png")))
    return tile_paths


def _collect_tiles_by_slide(tiles_dir: Path) -> dict[str, list[Path]]:
    """Group tile paths by slide name."""
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


def _estimate_pos_weight(
    tile_paths: list[Path],
    deconvolver: StainDeconvolver,
    dab_threshold: float,
    max_sample: int = 200,
    seed: int = 42,
) -> float:
    """Estimate the positive class weight for BCEWithLogitsLoss.

    Samples tiles, computes the membrane fraction, and returns
    neg_count / pos_count.
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
        total_pos += n_pos
        total_neg += int(membrane.size - n_pos)

    if total_pos == 0:
        logger.warning("No positive membrane pixels found; using pos_weight=1.0")
        return 1.0

    weight = total_neg / total_pos
    logger.info(
        "Estimated pos_weight: %.2f (%.1f%% membrane across %d tiles)",
        weight, 100.0 * total_pos / (total_pos + total_neg), len(sampled),
    )
    return weight


# ---------------------------------------------------------------------------
# Zero-shot inference (SAM/MedSAM automatic mask generator)
# ---------------------------------------------------------------------------


def _generate_point_prompts(
    dab: np.ndarray, dab_threshold: float, max_points: int = 32,
) -> np.ndarray:
    """Sample foreground point prompts from DAB-positive regions.

    Args:
        dab: DAB concentration map (H, W).
        dab_threshold: Threshold for positive regions.
        max_points: Maximum number of prompt points.

    Returns:
        Array of shape (N, 2) with (x, y) coordinates, or empty (0, 2).
    """
    ys, xs = np.where(dab > dab_threshold)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    indices = np.random.choice(len(ys), size=min(max_points, len(ys)), replace=False)
    return np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)


def zeroshot_membrane_map(
    rgb: np.ndarray,
    backend: str,
    device: torch.device,
    deconvolver: StainDeconvolver,
    dab_threshold: float = 0.10,
) -> np.ndarray:
    """Produce a membrane probability map via zero-shot SAM/MedSAM prompting.

    Uses automatic point prompts at DAB-positive locations to guide SAM
    toward membrane regions.

    Args:
        rgb: RGB tile (H, W, 3) uint8.
        backend: ``"medsam2"`` or ``"sam"``.
        device: Torch device.
        deconvolver: Stain deconvolver.
        dab_threshold: DAB threshold for prompt generation.

    Returns:
        Membrane probability map (H, W) float32 in [0, 1].
    """
    H, W = rgb.shape[:2]
    dab = deconvolver.extract_dab(rgb)
    points = _generate_point_prompts(dab, dab_threshold)

    if points.shape[0] == 0:
        logger.debug("No DAB-positive points; returning zeros")
        return np.zeros((H, W), dtype=np.float32)

    try:
        if backend == _BACKEND_MEDSAM2:
            sam_model = _load_medsam2_model(device)
        else:
            sam_model = _load_sam_model(device)

        from segment_anything import SamPredictor  # type: ignore[import-untyped]
        predictor = SamPredictor(sam_model)
        predictor.set_image(rgb)

        point_labels = np.ones(points.shape[0], dtype=np.int32)  # all foreground
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            multimask_output=True,
        )
        # Pick the mask with highest score
        best_idx = int(np.argmax(scores))
        prob_map = masks[best_idx].astype(np.float32)
        return prob_map

    except Exception as exc:
        logger.warning(
            "Zero-shot SAM inference failed (%s); returning DAB threshold as fallback",
            exc,
        )
        return (dab > dab_threshold).astype(np.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_membrane_medsam(
    tile_paths: list[Path],
    deconvolver: StainDeconvolver,
    save_path: Path,
    backend: str,
    dab_threshold: float = 0.10,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    patience: int = 5,
    val_fraction: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
    device_str: str = "cuda",
    img_size: int = 256,
    freeze_encoder: bool = True,
) -> nn.Module:
    """Fine-tune MedSAM 2 / SAM / ViT-UNet for membrane detection.

    The image encoder (ViT) is frozen; only the mask decoder is trained.
    Uses BCEWithLogitsLoss + DiceLoss combined, with early stopping.

    On CUDA OOM the batch size is halved automatically (down to 1).

    Args:
        tile_paths: All available tile paths.
        deconvolver: Stain deconvolver instance.
        save_path: Where to save the best checkpoint.
        backend: Model backend identifier.
        dab_threshold: DAB threshold for ground truth labels.
        epochs: Maximum training epochs.
        batch_size: Starting batch size (auto-reduced on OOM).
        lr: Learning rate.
        patience: Early stopping patience.
        val_fraction: Validation fraction.
        num_workers: DataLoader workers.
        seed: Random seed.
        device_str: Device string.
        img_size: Tile resize target.
        freeze_encoder: Freeze SAM/MedSAM encoder.

    Returns:
        The trained model loaded with best weights.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info("Training device: %s", device)

    # Build model
    model = build_model(backend, device, img_size=img_size, freeze_encoder=freeze_encoder)

    # Dataset
    full_dataset = MedSAMMembraneDataset(
        tile_paths, deconvolver, dab_threshold,
        target_size=img_size, augment=True,
    )
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    logger.info("Dataset split: %d train, %d val (%d total)", n_train, n_val, n_total)

    # Positive weight for class imbalance
    pos_weight_val = _estimate_pos_weight(tile_paths, deconvolver, dab_threshold, seed=seed)
    pos_weight = torch.tensor([pos_weight_val], device=device)

    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)

    # Only optimise trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state: Optional[dict[str, Any]] = None

    # Auto-reduce batch size on OOM
    current_batch_size = batch_size

    def _make_loaders(bs: int) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            train_dataset, batch_size=bs, shuffle=True,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
            drop_last=True, persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=bs, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )
        return train_loader, val_loader

    train_loader, val_loader = _make_loaders(current_batch_size)

    logger.info("Starting training: %d epochs, batch_size=%d, patience=%d", epochs, current_batch_size, patience)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            try:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(inputs)
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += loss.item()
                train_batches += 1

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_bs = max(1, current_batch_size // 2)
                if new_bs == current_batch_size:
                    logger.error("OOM with batch_size=1; cannot continue")
                    raise
                logger.warning(
                    "CUDA OOM: reducing batch_size %d -> %d",
                    current_batch_size, new_bs,
                )
                current_batch_size = new_bs
                train_loader, val_loader = _make_loaders(current_batch_size)
                break  # restart epoch with smaller batch

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
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %3d/%d  train_loss=%.5f  val_loss=%.5f  lr=%.2e  bs=%d",
            epoch, epochs, avg_train_loss, avg_val_loss, current_lr, current_batch_size,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("  -> New best val_loss=%.5f, saving checkpoint", best_val_loss)
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

    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "backend": backend, "img_size": img_size}, save_path)
        logger.info("Best model saved to %s (val_loss=%.5f)", save_path, best_val_loss)
    else:
        logger.warning("No checkpoint was saved (training may have been too short)")

    return model


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------


def generate_improved_masks(
    model: nn.Module,
    cfg: dict[str, Any],
    deconvolver: StainDeconvolver,
    tiles_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    device: torch.device,
    backend: str,
    mode: str,
    membrane_threshold: float = 0.5,
    batch_size: int = 4,
    dab_threshold: float = 0.10,
    img_size: int = 256,
) -> None:
    """Generate cell masks using the trained membrane detector or zero-shot mode.

    For each tile that has existing nucleus masks:
      1. Predict membrane probability (fine-tuned model or zero-shot SAM)
      2. Threshold to binary membrane
      3. Watershed with nucleus seeds to produce cell instances
      4. Save to output directory

    Args:
        model: Trained membrane detection model (ignored in zeroshot mode).
        cfg: Pipeline configuration.
        deconvolver: Stain deconvolver.
        tiles_dir: Root tiles directory.
        masks_dir: Existing masks directory.
        output_dir: Output directory for improved masks.
        device: Torch device.
        backend: Model backend identifier.
        mode: ``"finetune"`` or ``"zeroshot"``.
        membrane_threshold: Probability threshold for binary membrane.
        batch_size: Inference batch size.
        dab_threshold: DAB threshold (used for zero-shot prompts).
        img_size: Target image size for the model.
    """
    ws_cfg = cfg["watershed"]
    mpp: float = cfg["tile_extraction"].get("target_mpp", 0.5)
    max_cell_radius_px = ws_cfg["max_cell_radius_um"] / mpp

    slide_tiles = _collect_tiles_by_slide(tiles_dir)
    if not slide_tiles:
        logger.error("No tiles found under %s", tiles_dir)
        return

    if mode == "finetune":
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

        # Tiles with existing nucleus masks
        valid_tiles: list[tuple[Path, Path]] = []
        for tp in tile_paths:
            nuc_mask_path = slide_mask_dir / f"{tp.stem}_nuclei.tiff"
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

            batch_rgbs: list[np.ndarray] = []
            batch_indices: list[int] = []  # indices into batch that loaded OK

            for i, (tp, _) in enumerate(batch):
                bgr = cv2.imread(str(tp), cv2.IMREAD_COLOR)
                if bgr is None:
                    logger.warning("Could not read tile %s; skipping", tp)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                batch_rgbs.append(rgb)
                batch_indices.append(i)

            if not batch_rgbs:
                continue

            # Generate membrane maps
            if mode == "zeroshot":
                prob_maps: list[np.ndarray] = []
                for rgb in batch_rgbs:
                    pm = zeroshot_membrane_map(rgb, backend, device, deconvolver, dab_threshold)
                    prob_maps.append(pm)
            else:
                # Fine-tuned model inference
                batch_tensors: list[torch.Tensor] = []
                for rgb in batch_rgbs:
                    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                    t = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
                    batch_tensors.append(t)

                input_batch = torch.stack(batch_tensors).to(device, non_blocking=True)

                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(input_batch)
                probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H_model, W_model)

                prob_maps = []
                for j, rgb in enumerate(batch_rgbs):
                    pm = probs[j, 0]  # (H_model, W_model)
                    # Resize back to original tile resolution
                    if pm.shape[0] != rgb.shape[0] or pm.shape[1] != rgb.shape[1]:
                        pm = cv2.resize(pm, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                    prob_maps.append(pm)

            # Watershed for each tile
            for j, bi in enumerate(batch_indices):
                tp, nuc_mask_path = batch[bi]
                rgb = batch_rgbs[j]
                prob_map = prob_maps[j]

                membrane_binary = (prob_map > membrane_threshold).astype(np.uint8) * 255
                nucleus_labels = tifffile.imread(str(nuc_mask_path)).astype(np.int32)

                if nucleus_labels.max() == 0:
                    cell_path = slide_output_dir / f"{tp.stem}_cells.tiff"
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

                cell_path = slide_output_dir / f"{tp.stem}_cells.tiff"
                tifffile.imwrite(
                    str(cell_path),
                    cell_labels.astype(np.uint16),
                    compression="zlib",
                )

                total_cells += int(cell_labels.max())
                total_tiles += 1

        logger.info("Slide %s complete: %d tiles processed", slide_name, len(valid_tiles))

    logger.info(
        "Mask generation complete: %d tiles, %d total cell instances",
        total_tiles, total_cells,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse arguments, train / run inference, generate masks."""
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune MedSAM 2 (or SAM / ViT-UNet fallback) for DAB membrane "
            "detection and generate improved cell masks."
        ),
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training epochs (default: 20).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size (default: 4). Auto-reduced on OOM.",
    )
    parser.add_argument(
        "--skip-training", action="store_true", default=False,
        help="Skip training; use existing checkpoint or zero-shot mode.",
    )
    parser.add_argument(
        "--mode", type=str, choices=["finetune", "zeroshot"], default="finetune",
        help="'finetune' trains/uses fine-tuned model; 'zeroshot' uses SAM prompting.",
    )
    parser.add_argument(
        "--backend", type=str, choices=["medsam2", "sam", "vit_unet"], default=None,
        help="Force a specific backend (default: auto-detect in priority order).",
    )
    parser.add_argument(
        "--img-size", type=int, default=256,
        help="Tile resize target for training/inference (default: 256).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4).",
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
    output_dir = data_dir / "masks_medsam2"
    model_path = model_dir / "membrane_medsam2.pth"

    epochs = args.epochs if args.epochs is not None else 20
    batch_size = args.batch_size if args.batch_size is not None else 4
    seed: int = cfg["training"].get("seed", 42)
    num_workers: int = cfg["training"].get("num_workers", 4)
    device_str: str = cfg["nucleus_detection"].get("device", "cuda")
    dab_threshold = 0.10
    img_size = args.img_size

    # Stain deconvolver
    sc = cfg["stain_deconvolution"]
    stain_matrix = build_stain_matrix(
        hematoxylin=sc["hematoxylin"],
        dab=sc["dab"],
        residual=sc["residual"],
    )
    deconvolver = StainDeconvolver(stain_matrix)

    # Resolve backend
    backend = resolve_backend(args.backend)

    # Collect tiles
    tile_paths = _collect_all_tile_paths(tiles_dir)
    if not tile_paths:
        logger.error("No tiles found under %s", tiles_dir)
        sys.exit(1)
    logger.info("Found %d tiles across all slides", len(tile_paths))

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("MedSAM 2 Membrane Detection Pipeline")
    logger.info("=" * 72)
    logger.info("Backend         : %s", backend)
    logger.info("Mode            : %s", args.mode)
    logger.info("Image size      : %d", img_size)
    logger.info("Tiles available : %d", len(tile_paths))

    # ------------------------------------------------------------------
    # Training or loading
    # ------------------------------------------------------------------
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    model: Optional[nn.Module] = None

    if args.mode == "finetune":
        if args.skip_training:
            if not model_path.exists():
                logger.error("Cannot skip training: checkpoint not found at %s", model_path)
                sys.exit(1)

            logger.info("Loading existing checkpoint from %s", model_path)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            saved_backend = checkpoint.get("backend", backend)
            saved_img_size = checkpoint.get("img_size", img_size)
            logger.info("Checkpoint backend: %s, img_size: %d", saved_backend, saved_img_size)

            model = build_model(saved_backend, device, img_size=saved_img_size, freeze_encoder=True)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            logger.info("Epochs          : %d", epochs)
            logger.info("Batch size      : %d (auto-reduced on OOM)", batch_size)
            logger.info("Learning rate   : %.1e", args.lr)
            logger.info("DAB threshold   : %.2f", dab_threshold)
            logger.info("Early stopping  : patience=5")
            logger.info("=" * 72)

            model = train_membrane_medsam(
                tile_paths=tile_paths,
                deconvolver=deconvolver,
                save_path=model_path,
                backend=backend,
                dab_threshold=dab_threshold,
                epochs=epochs,
                batch_size=batch_size,
                lr=args.lr,
                patience=5,
                val_fraction=0.15,
                num_workers=num_workers,
                seed=seed,
                device_str=device_str,
                img_size=img_size,
                freeze_encoder=True,
            )
    elif args.mode == "zeroshot":
        if backend == _BACKEND_VIT_UNET:
            logger.error("Zero-shot mode requires SAM or MedSAM 2; ViT-UNet does not support it")
            sys.exit(1)
        logger.info("Zero-shot mode: no training needed; using SAM prompting")
        logger.info("=" * 72)

    # ------------------------------------------------------------------
    # Generate improved masks
    # ------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("Generating cell masks with %s (%s mode)", backend, args.mode)
    logger.info("=" * 72)

    generate_improved_masks(
        model=model if model is not None else nn.Identity(),
        cfg=cfg,
        deconvolver=deconvolver,
        tiles_dir=tiles_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        device=device,
        backend=backend,
        mode=args.mode,
        membrane_threshold=0.5,
        batch_size=batch_size,
        dab_threshold=dab_threshold,
        img_size=img_size,
    )

    logger.info("=" * 72)
    logger.info("All done.")
    logger.info("  Backend         : %s", backend)
    logger.info("  Mode            : %s", args.mode)
    if model_path.exists():
        logger.info("  Model checkpoint: %s", model_path)
    logger.info("  Masks output    : %s", output_dir)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
