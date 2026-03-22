#!/usr/bin/env python3
"""Train a 4-class membrane U-Net on bar-filter pseudo-labels.

Classes: 0=background, 1=nucleus, 2=cytoplasm, 3=membrane

The model learns to distinguish membrane from cytoplasm at the pixel level,
replacing the bar-filter with a learned approach that generalizes across
scanners without requiring stain-vector recalibration.

After training, validates by computing the membrane > cytoplasm DAB gap
on the test split.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/17_train_membrane_unet.py
"""
from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver

import sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stderr, force=True)
logger = logging.getLogger(__name__)
# Ensure all log output is flushed immediately
for h in logging.root.handlers:
    h.flush = lambda: sys.stderr.flush()

# ── Hyperparameters ───────────────────────────────────────────────────────
N_CLASSES = 4
CHANNELS = (32, 64, 128, 256)
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 50
PATIENCE = 50  # Run all epochs (user preference: no early stopping)
SEED = 42
NUM_WORKERS = 8  # 16 workers used 10GB RAM + swap thrashing; 8 is the sweet spot for 92GB system

# Class weights: up-weight membrane (rare) and nucleus, down-weight background
CLASS_WEIGHTS = torch.tensor([0.5, 2.0, 1.5, 4.0], dtype=torch.float32)

# Ring measurement parameters for validation
RING_ERODE_KSIZE = 5
RING_ERODE_ITER = 2


# ═══════════════════════════════════════════════════════════════════════════
# MODEL: 4-CLASS U-NET
# ═══════════════════════════════════════════════════════════════════════════

class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class MembraneUNet4C(nn.Module):
    """4-class U-Net: background / nucleus / cytoplasm / membrane."""

    def __init__(self, channels: tuple[int, ...] = CHANNELS, n_classes: int = N_CLASSES):
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
        self.outc = nn.Conv2d(c0, n_classes, kernel_size=1)

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


# ═══════════════════════════════════════════════════════════════════════════
# LOSS: CE + DICE
# ═══════════════════════════════════════════════════════════════════════════

class CEDiceLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, smooth: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        dice_sum = 0.0
        for c in range(logits.shape[1]):
            p = probs[:, c].reshape(-1)
            t = (targets == c).float().reshape(-1)
            inter = (p * t).sum()
            dice = (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_sum += (1.0 - dice)
        return ce_loss + dice_sum / logits.shape[1]


# ═══════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════

class MembraneDataset(Dataset):
    """Loads RGB tiles + 4-class label maps."""

    def __init__(self, tile_paths: list[Path], label_dir: Path, augment: bool = False):
        self.tile_paths = tile_paths
        self.label_dir = label_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int):
        tile_path = self.tile_paths[idx]
        slide = tile_path.parent.name
        stem = tile_path.stem

        # Load RGB tile
        rgb = cv2.imread(str(tile_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load label
        label_path = self.label_dir / slide / f"{stem}_label.png"
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

        if label is None:
            # Fallback: all background
            label = np.zeros(rgb.shape[:2], dtype=np.uint8)

        # Augmentation (geometric only — no stain aug per ablation finding)
        if self.augment:
            k = random.randint(0, 3)
            rgb = np.rot90(rgb, k).copy()
            label = np.rot90(label, k).copy()
            if random.random() > 0.5:
                rgb = np.fliplr(rgb).copy()
                label = np.fliplr(label).copy()
            if random.random() > 0.5:
                rgb = np.flipud(rgb).copy()
                label = np.flipud(label).copy()

        # To tensors
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3, H, W)
        label_t = torch.from_numpy(label.astype(np.int64))  # (H, W)

        return rgb_t, label_t


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for rgb, label in loader:
        rgb = rgb.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(rgb)
            loss = loss_fn(logits, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    class_correct = torch.zeros(N_CLASSES, device=device)
    class_total = torch.zeros(N_CLASSES, device=device)
    n_batches = 0

    for rgb, label in loader:
        rgb = rgb.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(rgb)
            loss = loss_fn(logits, label)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        for c in range(N_CLASSES):
            mask = label == c
            class_correct[c] += (pred[mask] == c).sum()
            class_total[c] += mask.sum()
        n_batches += 1

    per_class_acc = (class_correct / class_total.clamp(min=1)).cpu().numpy()
    return total_loss / max(n_batches, 1), per_class_acc


# ═══════════════════════════════════════════════════════════════════════════
# MEMBRANE GAP VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def validate_membrane_gap(model, test_paths, label_dir, deconv, device):
    """Run model on test tiles, use predicted membrane mask for DAB measurement."""
    model.eval()
    membrane_dabs = []
    cytoplasm_dabs = []
    n_cells = 0

    for tile_path in tqdm(test_paths[:100], desc="Gap validation"):
        rgb = cv2.imread(str(tile_path))
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Model prediction
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda"):
            logits = model(rgb_t)
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()  # (H, W)

        # DAB extraction
        dab = deconv.extract_dab(rgb)

        # Load cell instance labels (from masks)
        slide = tile_path.parent.name
        stem = tile_path.stem
        cell_path = tile_path.parent.parent.parent / "masks" / slide / f"{stem}_cells.tiff"
        if not cell_path.exists():
            continue
        cell_labels = tifffile.imread(str(cell_path))

        # Per-cell measurement using MODEL-PREDICTED membrane
        for cell_id in np.unique(cell_labels):
            if cell_id == 0:
                continue
            cell_mask = cell_labels == cell_id
            if cell_mask.sum() < 20:
                continue

            # Membrane = model predicts class 3 within this cell
            cell_membrane = cell_mask & (pred == 3)
            # Cytoplasm = model predicts class 2 within this cell
            cell_cyto = cell_mask & (pred == 2)

            if cell_membrane.sum() < 3 or cell_cyto.sum() < 3:
                continue

            membrane_dabs.append(float(dab[cell_membrane].mean()))
            cytoplasm_dabs.append(float(dab[cell_cyto].mean()))
            n_cells += 1

    if not membrane_dabs:
        return {"gap": 0, "n_cells": 0, "status": "NO DATA"}

    mem = float(np.mean(membrane_dabs))
    cyto = float(np.mean(cytoplasm_dabs))
    gap = mem - cyto
    return {
        "membrane_dab_mean": round(mem, 4),
        "cytoplasm_dab_mean": round(cyto, 4),
        "gap": round(gap, 4),
        "status": "PASS" if gap > 0 else "FAIL",
        "n_cells": n_cells,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility + performance
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True  # auto-tune convolution algorithms for fixed input size
    torch.set_float32_matmul_precision("high")  # use TF32 on Ampere

    cfg = load_config("config/default.yaml")
    data_dir = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    eval_dir = Path(cfg["paths"]["evaluation_dir"])
    label_dir = data_dir / "membrane_labels"
    sc = cfg["stain_deconvolution"]
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))

    # Check labels exist
    index_path = label_dir / "label_index.csv"
    if not index_path.exists():
        logger.error("Label index not found at %s — run 16_generate_membrane_labels.py first", index_path)
        return

    # Load index for weighted sampling
    membrane_fracs = {}
    with open(index_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['slide']}/{row['tile']}"
            membrane_fracs[key] = float(row["membrane_fraction"])

    logger.info("Loaded %d tile entries from label index", len(membrane_fracs))

    # Collect all tile paths that have labels
    all_tiles = []
    for slide_dir in sorted((data_dir / "tiles").iterdir()):
        if not slide_dir.is_dir():
            continue
        for tile_path in sorted(slide_dir.glob("*.png")):
            key = f"{slide_dir.name}/{tile_path.stem}"
            label_path = label_dir / slide_dir.name / f"{tile_path.stem}_label.png"
            if label_path.exists():
                all_tiles.append(tile_path)

    logger.info("Found %d tiles with labels", len(all_tiles))

    # Slide-level split (same as InstanSeg training)
    slides = sorted(set(p.parent.name for p in all_tiles))
    rng = np.random.default_rng(SEED)
    rng.shuffle(slides)
    n_train = int(len(slides) * 0.70)
    n_val = int(len(slides) * 0.15)
    train_slides = set(slides[:n_train])
    val_slides = set(slides[n_train:n_train + n_val])
    test_slides = set(slides[n_train + n_val:])

    train_tiles = [p for p in all_tiles if p.parent.name in train_slides]
    val_tiles = [p for p in all_tiles if p.parent.name in val_slides]
    test_tiles = [p for p in all_tiles if p.parent.name in test_slides]

    logger.info("Split: %d train / %d val / %d test (by slide: %d/%d/%d)",
                len(train_tiles), len(val_tiles), len(test_tiles),
                len(train_slides), len(val_slides), len(test_slides))

    # Weighted sampler: oversample membrane-rich tiles
    weights = []
    for p in train_tiles:
        key = f"{p.parent.name}/{p.stem}"
        frac = membrane_fracs.get(key, 0.0)
        weights.append(max(frac, 0.01))  # minimum weight for negative tiles
    sampler = WeightedRandomSampler(weights, num_samples=len(train_tiles), replacement=True)

    # Datasets
    train_ds = MembraneDataset(train_tiles, label_dir, augment=True)
    val_ds = MembraneDataset(val_tiles, label_dir, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )

    # Model — use both GPUs via DataParallel
    model = MembraneUNet4C(channels=CHANNELS, n_classes=N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_gpus = torch.cuda.device_count()
    # Single GPU is faster for this small model (7.7M params).
    # DataParallel overhead dominates: 35 min/epoch vs 20 min/epoch single-GPU.
    logger.info("MembraneUNet4C: %.2fM parameters — single GPU (DP overhead too high for small model)",
                n_params / 1e6)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = CEDiceLoss(class_weights=CLASS_WEIGHTS.to(device))

    # Training loop
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = 0

    logger.info("=" * 60)
    logger.info("TRAINING: %d epochs, batch=%d, lr=%.1e, patience=%d",
                EPOCHS, BATCH_SIZE, LR, PATIENCE)
    logger.info("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        t_epoch = time.time()

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device)
        val_loss, per_class_acc = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        elapsed_epoch = time.time() - t_epoch
        logger.info(
            "Epoch %3d/%d | train=%.4f  val=%.4f | acc: bg=%.2f nuc=%.2f cyto=%.2f mem=%.2f | lr=%.2e | %.1fs",
            epoch, EPOCHS, train_loss, val_loss,
            per_class_acc[0], per_class_acc[1], per_class_acc[2], per_class_acc[3],
            optimizer.param_groups[0]["lr"], elapsed_epoch,
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # Save best checkpoint (unwrap DataParallel if needed)
            ckpt_path = model_dir / "membrane_unet4c_best.pth"
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "per_class_acc": per_class_acc.tolist(),
            }, ckpt_path)
            logger.info("  -> Best model saved (epoch %d, val_loss=%.4f)", epoch, val_loss)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                logger.info("Early stopping at epoch %d (best=%d)", epoch, best_epoch)
                break

    # Load best checkpoint for validation (into single-GPU model for inference)
    ckpt = torch.load(model_dir / "membrane_unet4c_best.pth", weights_only=False)
    val_model = MembraneUNet4C(channels=CHANNELS, n_classes=N_CLASSES).to(device)
    val_model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded best checkpoint from epoch %d", ckpt["epoch"])
    model = val_model  # use single-GPU model for gap validation

    # ── MEMBRANE GAP VALIDATION ──
    logger.info("=" * 60)
    logger.info("MEMBRANE GAP VALIDATION on test split")
    logger.info("=" * 60)

    import tifffile  # needed for gap validation
    gap_result = validate_membrane_gap(model, test_tiles, label_dir, deconv, device)

    logger.info("Membrane DAB: %.4f", gap_result.get("membrane_dab_mean", 0))
    logger.info("Cytoplasm DAB: %.4f", gap_result.get("cytoplasm_dab_mean", 0))
    logger.info("Gap: %+.4f", gap_result["gap"])
    logger.info("Status: %s", gap_result["status"])
    logger.info("Cells: %d", gap_result["n_cells"])
    logger.info("-" * 60)
    logger.info("REFERENCE: bar-filter gap = +0.1269")
    logger.info("REFERENCE: fixed-ring gap = -0.0201")

    # Save results
    results = {
        "training": {
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "per_class_acc": ckpt["per_class_acc"],
            "n_params": n_params,
            "n_train_tiles": len(train_tiles),
            "n_val_tiles": len(val_tiles),
            "n_test_tiles": len(test_tiles),
        },
        "membrane_gap_validation": gap_result,
        "total_time_seconds": round(time.time() - t0, 1),
    }
    output_path = eval_dir / "membrane_unet_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    elapsed = time.time() - t0
    logger.info("Total time: %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
