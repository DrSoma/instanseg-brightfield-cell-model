#!/usr/bin/env python3
"""MEGAZORD PIPELINE: Iterative self-training → Wider U-Net → InstanSeg v3.0

Three compounding stages:
  Stage 1: Generate Gen-2 labels from current U-Net's softmax predictions
  Stage 2: Train Gen-2 Wider U-Net (48,96,192,384 + attention) on Gen-1 soft labels
  Stage 3: Use Gen-2 to generate membrane-aligned cell masks → retrain InstanSeg

Usage:
    PYTHONUNBUFFERED=1 python scripts/19_megazord_pipeline.py 2>&1 | tee /tmp/megazord.log
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
    force=True,
)
logger = logging.getLogger("megazord")

PROJECT = Path("/home/fernandosoto/Documents/instanseg-brightfield-cell-model")
sys.path.insert(0, str(PROJECT))
from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True       # auto-tune conv algorithms for fixed input
torch.backends.cudnn.deterministic = False   # allow non-deterministic for speed
torch.set_float32_matmul_precision("high")   # TF32 on Ampere
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Prevent swap thrashing: use 'spawn' instead of 'fork' for DataLoader workers.
# Fork causes copy-on-write memory duplication that fills swap over time.
import torch.multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections — focuses decoder on membrane pixels."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False), nn.Sigmoid())
        self.relu = nn.ReLU(True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        # Upsample gate to match skip spatial dims
        g = F.interpolate(g, size=s.shape[2:], mode="bilinear", align_corners=False)
        attn = self.psi(self.relu(g + s))
        return skip * attn


class _UpAttn(nn.Module):
    """Decoder block with attention gate on skip connection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.attn = AttentionGate(gate_ch=in_ch // 2, skip_ch=in_ch // 2, inter_ch=in_ch // 4)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        skip = self.attn(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class _Up(nn.Module):
    """Standard decoder block (no attention)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class MembraneUNet4C(nn.Module):
    """Gen-1: Standard 4-class U-Net (32,64,128,256)."""

    def __init__(self, channels=(32, 64, 128, 256), n_classes=4):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.inc = _DoubleConv(3, c0)
        self.down1, self.down2, self.down3 = _Down(c0, c1), _Down(c1, c2), _Down(c2, c3)
        self.bottleneck = _Down(c3, c3 * 2)
        self.up1, self.up2, self.up3, self.up4 = _Up(c3*2, c3), _Up(c3, c2), _Up(c2, c1), _Up(c1, c0)
        self.outc = nn.Conv2d(c0, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.bottleneck(x4)
        return self.outc(self.up4(self.up3(self.up2(self.up1(x5, x4), x3), x2), x1))


class WiderAttnUNet(nn.Module):
    """Gen-2: Wider channels (48,96,192,384) + attention gates on skip connections."""

    def __init__(self, channels=(48, 96, 192, 384), n_classes=4):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.inc = _DoubleConv(3, c0)
        self.down1, self.down2, self.down3 = _Down(c0, c1), _Down(c1, c2), _Down(c2, c3)
        self.bottleneck = _Down(c3, c3 * 2)
        # Attention gates on skip connections
        self.up1 = _UpAttn(c3 * 2, c3)
        self.up2 = _UpAttn(c3, c2)
        self.up3 = _UpAttn(c2, c1)
        self.up4 = _UpAttn(c1, c0)
        self.outc = nn.Conv2d(c0, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.bottleneck(x4)
        return self.outc(self.up4(self.up3(self.up2(self.up1(x5, x4), x3), x2), x1))


# ═══════════════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════════════


class CEDiceLoss(nn.Module):
    def __init__(self, class_weights, smooth=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.smooth = smooth

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        dice_sum = 0.0
        for c in range(logits.shape[1]):
            p, t = probs[:, c].reshape(-1), (targets == c).float().reshape(-1)
            dice_sum += 1.0 - (2 * (p * t).sum() + self.smooth) / (p.sum() + t.sum() + self.smooth)
        return ce + dice_sum / logits.shape[1]


class SoftCEDiceLoss(nn.Module):
    """Loss for soft labels (probabilities, not hard classes)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, soft_targets):
        """logits: (B, C, H, W), soft_targets: (B, C, H, W) probabilities."""
        log_probs = F.log_softmax(logits, dim=1)
        ce = -(soft_targets * log_probs).sum(dim=1).mean()

        probs = torch.softmax(logits, dim=1)
        dice_sum = 0.0
        for c in range(logits.shape[1]):
            p, t = probs[:, c].reshape(-1), soft_targets[:, c].reshape(-1)
            dice_sum += 1.0 - (2 * (p * t).sum() + self.smooth) / (p.sum() + t.sum() + self.smooth)
        return ce + dice_sum / logits.shape[1]


# ═══════════════════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════════════════


class HardLabelDataset(Dataset):
    """Loads RGB + hard uint8 label maps (for Gen-1 style training)."""

    def __init__(self, tile_paths, label_dir, augment=False):
        self.tile_paths, self.label_dir, self.augment = tile_paths, label_dir, augment

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        p = self.tile_paths[idx]
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(str(self.label_dir / p.parent.name / f"{p.stem}_label.png"), cv2.IMREAD_GRAYSCALE)
        if lbl is None:
            lbl = np.zeros(rgb.shape[:2], dtype=np.uint8)
        if self.augment:
            k = random.randint(0, 3); rgb = np.rot90(rgb, k).copy(); lbl = np.rot90(lbl, k).copy()
            if random.random() > 0.5: rgb = np.fliplr(rgb).copy(); lbl = np.fliplr(lbl).copy()
        return (torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1),
                torch.from_numpy(lbl.astype(np.int64)))


class SoftLabelDataset(Dataset):
    """Loads RGB + soft 4-channel probability maps (for Gen-2 training)."""

    def __init__(self, tile_paths, soft_label_dir, augment=False):
        self.tile_paths, self.soft_label_dir, self.augment = tile_paths, soft_label_dir, augment

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        p = self.tile_paths[idx]
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        # Prefer .npy (instant load) over .npz (requires decompression)
        npy_path = self.soft_label_dir / p.parent.name / f"{p.stem}_soft.npy"
        npz_path = self.soft_label_dir / p.parent.name / f"{p.stem}_soft.npz"
        if npy_path.exists():
            soft = np.load(str(npy_path))  # (4, H, W) float16, instant
        elif npz_path.exists():
            soft = np.load(str(npz_path))["probs"]  # fallback to compressed
        else:
            soft = np.zeros((4, rgb.shape[0], rgb.shape[1]), dtype=np.float32)
            soft[0] = 1.0
        if self.augment:
            k = random.randint(0, 3)
            rgb = np.rot90(rgb, k).copy()
            soft = np.rot90(soft, k, axes=(1, 2)).copy()
            if random.random() > 0.5:
                rgb = np.fliplr(rgb).copy()
                soft = np.flip(soft, axis=2).copy()
        return (torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1),
                torch.from_numpy(soft.astype(np.float32)))


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════


def train_model(
    model, train_loader, val_loader, loss_fn, device,
    epochs=999, lr=1e-3, patience=20, save_path=None, soft_labels=False,
):
    """Generic training loop for both hard and soft label training."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Use ReduceLROnPlateau — drops LR when val_loss plateaus, better than cosine for indefinite training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-7, verbose=True,
    )
    scaler = torch.amp.GradScaler("cuda")
    best_vl, best_ep = float("inf"), 0
    epochs_no_improve = 0
    start_epoch = 1

    # Resume from checkpoint if it exists and has optimizer state
    if save_path and Path(save_path).exists():
        ckpt = torch.load(save_path, weights_only=False, map_location=device)
        if "optimizer_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_vl = ckpt.get("val_loss", float("inf"))
            best_ep = ckpt.get("best_epoch", ckpt.get("epoch", 0))
            start_epoch = ckpt.get("epoch", 0) + 1
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            logger.info("Resumed from epoch %d (val_loss=%.4f, lr=%.1e)",
                        start_epoch - 1, best_vl, optimizer.param_groups[0]["lr"])
        else:
            logger.info("Checkpoint exists but no optimizer state — training from scratch")

    for ep in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, n = 0.0, 0
        for rgb, lbl in train_loader:
            rgb = rgb.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(rgb)
                loss = loss_fn(logits, lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n += 1

        model.eval()
        val_loss, vn = 0.0, 0
        cc, ct = torch.zeros(4, device=device), torch.zeros(4, device=device)
        with torch.inference_mode():
            for rgb, lbl in val_loader:
                rgb = rgb.to(device, non_blocking=True)
                lbl = lbl.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(rgb)
                    loss = loss_fn(logits, lbl)
                val_loss += loss.item()
                vn += 1
                pred = logits.argmax(dim=1)
                tgt = lbl.argmax(dim=1) if soft_labels else lbl
                for c in range(4):
                    m = tgt == c
                    cc[c] += (pred[m] == c).sum()
                    ct[c] += m.sum()

        vl = val_loss / max(vn, 1)
        scheduler.step(vl)  # ReduceLROnPlateau uses val_loss
        acc = (cc / ct.clamp(min=1)).cpu().numpy()
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Ep %3d | t=%.4f v=%.4f | bg=%.2f nu=%.2f cy=%.2f me=%.2f | lr=%.1e | %.0fs",
            ep, train_loss / max(n, 1), vl,
            acc[0], acc[1], acc[2], acc[3], current_lr, time.time() - t0,
        )

        if vl < best_vl and save_path:
            best_vl, best_ep = vl, ep
            epochs_no_improve = 0
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": ep,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": vl,
                "per_class_acc": acc.tolist(),
                "best_epoch": best_ep,
                "epochs_no_improve": epochs_no_improve,
            }, save_path)
            logger.info("  -> Best (ep %d, vl=%.4f)", ep, vl)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at ep %d — no improvement for %d epochs (best: ep %d, vl=%.4f)",
                            ep, patience, best_ep, best_vl)
                break

    return best_ep, best_vl


# ═══════════════════════════════════════════════════════════════════════════
# GAP VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


@torch.inference_mode()
def validate_gap(model, data_dir, device, n_tiles=200):
    """Measure membrane > cytoplasm DAB gap using model predictions."""
    cfg = load_config(str(PROJECT / "config" / "default.yaml"))
    sc = cfg["stain_deconvolution"]
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))
    ds = torch.load(data_dir / "segmentation_dataset_base.pth", weights_only=False)
    test_data = ds["Test"][:n_tiles]

    argmax_mem, argmax_cyto, soft_mem, soft_cyto = [], [], [], []

    for item in tqdm(test_data, desc="Gap validation"):
        img = tifffile.imread(str(data_dir / item["image"]))
        if img is None:
            continue
        dab = deconv.extract_dab(img)
        rgb_t = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.amp.autocast("cuda"):
            logits = model(rgb_t)
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        p = Path(item["image"])
        cell_path = data_dir / "masks" / p.parts[1] / f"{p.stem}_cells.tiff"
        if not cell_path.exists():
            continue
        cells = tifffile.imread(str(cell_path))

        for cid in np.unique(cells):
            if cid == 0:
                continue
            mask = cells == cid
            if mask.sum() < 20:
                continue

            cell_mem = mask & (pred == 3)
            cell_cyto = mask & (pred == 2)
            if cell_mem.sum() >= 3 and cell_cyto.sum() >= 3:
                argmax_mem.append(float(dab[cell_mem].mean()))
                argmax_cyto.append(float(dab[cell_cyto].mean()))

            mw, cw, dv = probs[3][mask], probs[2][mask], dab[mask]
            if mw.sum() > 0.1 and cw.sum() > 0.1:
                soft_mem.append(float(np.average(dv, weights=np.clip(mw, 0, None))))
                soft_cyto.append(float(np.average(dv, weights=np.clip(cw, 0, None))))

    results = {}
    if argmax_mem:
        m, c = np.mean(argmax_mem), np.mean(argmax_cyto)
        g = m - c
        results["argmax"] = {"membrane": round(m, 4), "cytoplasm": round(c, 4),
                             "gap": round(g, 4), "status": "PASS" if g > 0 else "FAIL",
                             "n_cells": len(argmax_mem)}
        logger.info("ARGMAX: mem=%.4f cyto=%.4f gap=%+.4f %s (%d cells)",
                    m, c, g, "PASS" if g > 0 else "FAIL", len(argmax_mem))

    if soft_mem:
        m, c = np.mean(soft_mem), np.mean(soft_cyto)
        g = m - c
        results["soft"] = {"membrane": round(m, 4), "cytoplasm": round(c, 4),
                           "gap": round(g, 4), "status": "PASS" if g > 0 else "FAIL",
                           "n_cells": len(soft_mem)}
        logger.info("SOFT:   mem=%.4f cyto=%.4f gap=%+.4f %s (%d cells)",
                    m, c, g, "PASS" if g > 0 else "FAIL", len(soft_mem))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: Generate Gen-2 soft labels from Gen-1
# ═══════════════════════════════════════════════════════════════════════════


def stage1_generate_soft_labels(gen1_path: Path, data_dir: Path, output_dir: Path, device: str):
    """Run Gen-1 U-Net on all tiles, save softmax probabilities as training labels for Gen-2."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Generating Gen-2 soft labels from Gen-1 predictions")
    logger.info("=" * 60)

    model = MembraneUNet4C().to(device)
    ckpt = torch.load(gen1_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Gen-1 loaded (epoch %d, val_loss=%.4f)", ckpt["epoch"], ckpt["val_loss"])

    output_dir.mkdir(parents=True, exist_ok=True)
    tile_dirs = sorted([d for d in (data_dir / "tiles").iterdir() if d.is_dir()])

    total = 0
    for slide_dir in tqdm(tile_dirs, desc="Stage 1: Soft labels"):
        out_dir = output_dir / slide_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        tiles = sorted(slide_dir.glob("*.tiff")) + sorted(slide_dir.glob("*.png"))
        batch_imgs, batch_stems = [], []

        for tile_path in tiles:
            soft_path = out_dir / f"{tile_path.stem}_soft.npz"
            if soft_path.exists():
                total += 1
                continue

            img = tifffile.imread(str(tile_path)) if tile_path.suffix == ".tiff" else cv2.cvtColor(
                cv2.imread(str(tile_path)), cv2.COLOR_BGR2RGB)
            if img is None:
                continue
            batch_imgs.append(img)
            batch_stems.append(tile_path.stem)

            if len(batch_imgs) >= 64:  # larger batches for GPU throughput
                _process_soft_batch(model, batch_imgs, batch_stems, out_dir, device)
                total += len(batch_imgs)
                batch_imgs.clear()
                batch_stems.clear()

        if batch_imgs:
            _process_soft_batch(model, batch_imgs, batch_stems, out_dir, device)
            total += len(batch_imgs)
            batch_imgs.clear()
            batch_stems.clear()

    logger.info("Stage 1 complete: %d soft labels generated", total)
    return total


def _process_soft_batch(model, imgs, stems, out_dir, device):
    """Process a batch of images through the model and save soft labels."""
    batch = torch.stack([
        torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)
        for img in imgs
    ]).to(device)

    with torch.inference_mode(), torch.amp.autocast("cuda"):
        logits = model(batch)
    probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B, 4, H, W)

    for i, stem in enumerate(stems):
        np.savez_compressed(
            str(out_dir / f"{stem}_soft.npz"),
            probs=probs[i].astype(np.float16),  # Save as fp16 to reduce storage
        )


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: Train Gen-2 Wider Attention U-Net
# ═══════════════════════════════════════════════════════════════════════════


def stage2_train_gen2(data_dir: Path, soft_label_dir: Path, save_path: Path, device: str):
    """Train Gen-2 wider U-Net with attention on Gen-1 soft labels."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Training Gen-2 Wider Attention U-Net")
    logger.info("=" * 60)

    # Use NVMe paths if available (10-50x faster I/O than HDD)
    nvme_tiles = Path("/opt/training_data/tiles")
    nvme_labels = Path("/opt/training_data/membrane_soft_labels")
    if nvme_tiles.exists() and nvme_labels.exists():
        tile_base = nvme_tiles
        label_base = nvme_labels
        logger.info("Using NVMe storage for tiles and labels (fast I/O)")
    else:
        tile_base = data_dir / "tiles"
        label_base = soft_label_dir
        logger.info("Using default storage (HDD)")

    # Collect tiles with soft labels
    tiles = []
    for slide_dir in sorted(tile_base.iterdir()):
        if not slide_dir.is_dir():
            continue
        for tp in sorted(slide_dir.glob("*.tiff")) + sorted(slide_dir.glob("*.png")):
            soft_path = label_base / slide_dir.name / f"{tp.stem}_soft.npy"
            if not soft_path.exists():
                soft_path = label_base / slide_dir.name / f"{tp.stem}_soft.npz"
            if soft_path.exists():
                tiles.append(tp)

    # Slide-level split
    slides = sorted(set(p.parent.name for p in tiles))
    rng = np.random.default_rng(SEED)
    rng.shuffle(slides)
    nt, nv = int(len(slides) * 0.7), int(len(slides) * 0.15)
    tr_s, va_s = set(slides[:nt]), set(slides[nt:nt + nv])
    train_tiles = [p for p in tiles if p.parent.name in tr_s]
    val_tiles = [p for p in tiles if p.parent.name in va_s]
    logger.info("Split: %d train / %d val tiles", len(train_tiles), len(val_tiles))

    train_ds = SoftLabelDataset(train_tiles, label_base, augment=True)
    val_ds = SoftLabelDataset(val_tiles, label_base, augment=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True,
                              persistent_workers=True, prefetch_factor=8)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=8, pin_memory=True,
                            persistent_workers=True, prefetch_factor=8)

    model = WiderAttnUNet(channels=(48, 96, 192, 384)).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # DataParallel not compatible with AttentionGate's F.interpolate across GPUs
    # (causes CUDA illegal memory access). Single GPU is also faster for this model size.
    logger.info("WiderAttnUNet: %.2fM params (batch=64, prefetch=8, single GPU)", n_params / 1e6)

    loss_fn = SoftCEDiceLoss()
    best_ep, best_vl = train_model(
        model, train_loader, val_loader, loss_fn, device,
        epochs=999, lr=5e-4, patience=20, save_path=save_path, soft_labels=True,
    )
    logger.info("Stage 2 complete: best epoch %d, val_loss=%.4f", best_ep, best_vl)
    return best_ep, best_vl


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: Generate membrane-aligned masks for InstanSeg retraining
# ═══════════════════════════════════════════════════════════════════════════


def stage3_generate_aligned_masks(gen2_path: Path, data_dir: Path, output_dir: Path, device: str):
    """Use Gen-2 to refine cell masks: snap boundaries to predicted membrane pixels."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Generating membrane-aligned cell masks")
    logger.info("=" * 60)

    model = WiderAttnUNet(channels=(48, 96, 192, 384)).to(device)
    ckpt = torch.load(gen2_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Gen-2 loaded (epoch %d)", ckpt["epoch"])

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dirs = sorted([d for d in (data_dir / "masks").iterdir() if d.is_dir()])

    total_refined = 0
    for slide_dir in tqdm(mask_dirs, desc="Stage 3: Aligning masks"):
        out_dir = output_dir / slide_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for cell_path in sorted(slide_dir.glob("*_cells.tiff")):
            stem = cell_path.stem.replace("_cells", "")
            out_path = out_dir / f"{stem}_cells_aligned.tiff"
            if out_path.exists():
                total_refined += 1
                continue

            nuc_path = slide_dir / f"{stem}_nuclei.tiff"
            # Find tile (could be .tiff or .png)
            tile_path = data_dir / "tiles" / slide_dir.name / f"{stem}.tiff"
            if not tile_path.exists():
                tile_path = data_dir / "tiles" / slide_dir.name / f"{stem}.png"
            if not tile_path.exists() or not nuc_path.exists():
                continue

            # Load data
            cells = tifffile.imread(str(cell_path))
            nuclei = tifffile.imread(str(nuc_path))
            img = tifffile.imread(str(tile_path)) if tile_path.suffix == ".tiff" else cv2.cvtColor(
                cv2.imread(str(tile_path)), cv2.COLOR_BGR2RGB)

            # Model prediction
            rgb_t = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.inference_mode(), torch.amp.autocast("cuda"):
                logits = model(rgb_t)
            membrane_prob = torch.softmax(logits, dim=1)[0, 3].cpu().numpy()  # class 3 = membrane

            # Refine: for each cell, shrink boundary to membrane zone
            refined = _refine_masks_with_membrane(cells, nuclei, membrane_prob)

            tifffile.imwrite(str(out_path), refined.astype(np.uint16), compression="zlib")
            total_refined += 1

    logger.info("Stage 3 complete: %d masks refined", total_refined)
    return total_refined


def _refine_masks_with_membrane(
    cells: np.ndarray, nuclei: np.ndarray, membrane_prob: np.ndarray,
    membrane_threshold: float = 0.3,
) -> np.ndarray:
    """Refine cell masks: snap outer boundary to where membrane probability is high.

    For each cell, remove boundary pixels where membrane_prob < threshold.
    This shrinks cells to end at the predicted membrane.
    """
    from skimage.segmentation import watershed

    H, W = cells.shape
    refined = cells.copy()

    # Create a membrane-aware energy for watershed
    # High membrane probability = low energy = preferred boundary
    energy = 1.0 - membrane_prob

    # Use nuclei as seeds, cells > 0 as mask
    markers = nuclei.copy().astype(np.int32)
    mask = cells > 0

    # Re-run watershed with membrane-aware energy
    if markers.max() > 0 and mask.any():
        refined = watershed(energy, markers=markers, mask=mask, compactness=0.0)

    # Ensure nuclei are contained
    for nid in np.unique(nuclei):
        if nid > 0:
            refined[nuclei == nid] = nid

    return refined.astype(np.uint16)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: THE MEGAZORD
# ═══════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    device = "cuda"

    cfg = load_config(str(PROJECT / "config" / "default.yaml"))
    data_dir = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    eval_dir = Path(cfg["paths"]["evaluation_dir"])

    gen1_path = model_dir / "membrane_unet4c_best.pth"
    gen2_path = model_dir / "membrane_gen2_wider_attn_best.pth"
    soft_label_dir = data_dir / "membrane_soft_labels"
    aligned_mask_dir = data_dir / "masks_membrane_aligned"

    logger.info("=" * 60)
    logger.info("MEGAZORD PIPELINE: Gen-1 → Gen-2 → Aligned Masks")
    logger.info("=" * 60)

    all_results = {}

    # ── STAGE 1: Generate soft labels ──
    n_soft = stage1_generate_soft_labels(gen1_path, data_dir, soft_label_dir, device)
    all_results["stage1"] = {"n_soft_labels": n_soft}

    # ── STAGE 2: Train Gen-2 ──
    best_ep, best_vl = stage2_train_gen2(data_dir, soft_label_dir, gen2_path, device)
    all_results["stage2"] = {"best_epoch": best_ep, "best_val_loss": round(best_vl, 4)}

    # Validate Gen-2
    logger.info("Validating Gen-2...")
    model = WiderAttnUNet(channels=(48, 96, 192, 384)).to(device)
    ckpt = torch.load(gen2_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    gen2_gap = validate_gap(model, data_dir, device)
    all_results["gen2_gap"] = gen2_gap
    del model
    torch.cuda.empty_cache()

    # ── STAGE 3: Generate membrane-aligned masks ──
    n_aligned = stage3_generate_aligned_masks(gen2_path, data_dir, aligned_mask_dir, device)
    all_results["stage3"] = {"n_aligned_masks": n_aligned}

    # ── SAVE RESULTS ──
    all_results["total_time_minutes"] = round((time.time() - t0) / 60, 1)
    results_path = eval_dir / "megazord_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("=" * 60)
    logger.info("MEGAZORD COMPLETE in %.1f minutes", (time.time() - t0) / 60)
    logger.info("Results: %s", results_path)
    logger.info("Gen-2 checkpoint: %s", gen2_path)
    logger.info("Aligned masks: %s", aligned_mask_dir)
    logger.info("=" * 60)
    logger.info("")
    logger.info("NEXT: Retrain InstanSeg v3.0 on aligned masks:")
    logger.info("  1. Rebuild dataset: python scripts/03_prepare_dataset.py --mask-dir %s", aligned_mask_dir)
    logger.info("  2. Train: python scripts/04_train_v2d_zero_stain_aug.py")
    logger.info("  3. Validate: python scripts/07_validate_boundaries.py")


if __name__ == "__main__":
    main()
