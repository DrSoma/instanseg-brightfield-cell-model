#!/usr/bin/env python3
"""Gen-3 Multi-Vector Training: Scanner/staining robustness.

Generates bar-filter labels with MULTIPLE stain vector sets, then trains
Gen-3 on the combined labels. The model learns what's consistent across
all vector definitions (actual membrane) and ignores what's inconsistent
(vector-specific artifacts).

Stages:
  1. Generate bar-filter labels with 7 different stain vector sets
  2. For each tile, create an AVERAGED soft label across all vector sets
  3. Train Gen-3 WiderAttnUNet on the averaged labels
  4. Validate gap

Usage:
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/21_gen3_multivector.py > /tmp/gen3.log 2>&1 &
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stderr, force=True)
logger = logging.getLogger("gen3")

PROJECT = Path("/home/fernandosoto/Documents/instanseg-brightfield-cell-model")
sys.path.insert(0, str(PROJECT))
from instanseg_brightfield.config import load_config

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)

# ═══════════════════════════════════════════════════════════════════════════
# STAIN VECTOR SETS
# ═══════════════════════════════════════════════════════════════════════════

VECTOR_SETS = {
    "calibrated": {
        "hematoxylin": [0.786, 0.593, 0.174],
        "dab": [0.215, 0.422, 0.881],
        "residual": [0.547, -0.799, 0.249],
    },
    "ruifrok_default": {
        "hematoxylin": [0.650, 0.704, 0.286],
        "dab": [0.268, 0.570, 0.776],
        "residual": [0.711, 0.424, 0.562],
    },
    "interp_25": {  # 25% toward default
        "hematoxylin": [0.752, 0.621, 0.202],
        "dab": [0.228, 0.459, 0.855],
        "residual": [0.588, -0.493, 0.327],
    },
    "interp_50": {  # 50% midpoint
        "hematoxylin": [0.718, 0.649, 0.230],
        "dab": [0.242, 0.496, 0.829],
        "residual": [0.629, -0.188, 0.406],
    },
    "interp_75": {  # 75% toward default
        "hematoxylin": [0.684, 0.676, 0.258],
        "dab": [0.255, 0.533, 0.802],
        "residual": [0.670, 0.118, 0.484],
    },
    "perturb_warm": {  # Slightly warmer DAB (more red absorption)
        "hematoxylin": [0.770, 0.610, 0.190],
        "dab": [0.280, 0.440, 0.850],
        "residual": [0.530, -0.700, 0.280],
    },
    "perturb_cool": {  # Slightly cooler DAB (more blue absorption)
        "hematoxylin": [0.800, 0.575, 0.160],
        "dab": [0.190, 0.400, 0.900],
        "residual": [0.560, -0.850, 0.220],
    },
}


class SoftDS(Dataset):
    """Soft-label dataset at module level for spawn multiprocessing compatibility."""
    def __init__(self, paths, label_dir, augment=False):
        self.paths, self.label_dir, self.augment = paths, label_dir, augment
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        if p.suffix == ".tiff":
            rgb = tifffile.imread(str(p))
        else:
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        npy_path = self.label_dir / p.parent.name / f"{p.stem}_soft.npy"
        soft = np.load(str(npy_path)) if npy_path.exists() else np.zeros((4, rgb.shape[0], rgb.shape[1]), dtype=np.float32)
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
# MODEL + LOSS (module level for spawn multiprocessing compatibility)
# ═══════════════════════════════════════════════════════════════════════════

class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
    def forward(self, x): return self.block(x)

class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.pool_conv(x)

class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False), nn.Sigmoid())
        self.relu = nn.ReLU(True)
    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        g = F.interpolate(g, size=s.shape[2:], mode="bilinear", align_corners=False)
        return skip * self.psi(self.relu(g + s))

class _UpAttn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.attn = AttentionGate(in_ch // 2, in_ch // 2, in_ch // 4)
        self.conv = _DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        skip = self.attn(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))

class WiderAttnUNet(nn.Module):
    def __init__(self, channels=(48, 96, 192, 384), n_classes=4):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.inc = _DoubleConv(3, c0)
        self.down1, self.down2, self.down3 = _Down(c0, c1), _Down(c1, c2), _Down(c2, c3)
        self.bottleneck = _Down(c3, c3 * 2)
        self.up1, self.up2, self.up3, self.up4 = _UpAttn(c3*2, c3), _UpAttn(c3, c2), _UpAttn(c2, c1), _UpAttn(c1, c0)
        self.outc = nn.Conv2d(c0, n_classes, 1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.bottleneck(x4)
        return self.outc(self.up4(self.up3(self.up2(self.up1(x5, x4), x3), x2), x1))

class SoftCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, soft_targets):
        log_probs = F.log_softmax(logits, dim=1)
        ce = -(soft_targets * log_probs).sum(dim=1).mean()
        probs = torch.softmax(logits, dim=1)
        dice_sum = 0.0
        for c in range(logits.shape[1]):
            p, t = probs[:, c].reshape(-1), soft_targets[:, c].reshape(-1)
            dice_sum += 1.0 - (2 * (p * t).sum() + self.smooth) / (p.sum() + t.sum() + self.smooth)
        return ce + dice_sum / logits.shape[1]


def build_stain_inverse(h_vec, dab_vec, res_vec):
    """Build the inverse stain matrix for deconvolution."""
    matrix = np.array([h_vec, dab_vec, res_vec], dtype=np.float64)
    # Row-normalize
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms
    return np.linalg.inv(matrix).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# BAR-FILTER (reused from existing scripts)
# ═══════════════════════════════════════════════════════════════════════════

def build_bar_kernels(n=8, ksize=25, sigma_long=5.0, sigma_short=1.0):
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    kernels = []
    for i in range(n):
        theta = i * np.pi / n
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t
        g = np.exp(-0.5 * (x_rot**2 / sigma_long**2 + y_rot**2 / sigma_short**2))
        g -= g.mean()
        g /= np.abs(g).sum() + 1e-10
        kernels.append(g)
    return torch.from_numpy(np.stack(kernels)[:, np.newaxis, :, :]).float()


def extract_dab_with_vectors(tile_rgb, inv_matrix):
    """Extract DAB channel using given inverse stain matrix."""
    rgb_float = tile_rgb.astype(np.float32) / 255.0
    od = -np.log(np.clip(rgb_float, 1 / 255.0, 1.0))
    od_flat = od.reshape(-1, 3)
    conc = np.clip(od_flat @ inv_matrix, 0, None)
    return conc[:, 1].reshape(tile_rgb.shape[:2]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: Generate multi-vector averaged labels
# ═══════════════════════════════════════════════════════════════════════════

def stage1_generate_multivector_labels(data_dir: Path, output_dir: Path, device: str):
    """Generate 4-class labels averaged across 7 stain vector sets.

    For each tile:
    1. Extract DAB with each of 7 vector sets
    2. Run bar-filter on each DAB channel
    3. Generate 4-class label from each (using cell/nucleus masks)
    4. Average the membrane probabilities across all 7
    5. Save as a soft label (float16 npy)
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Multi-vector label generation (7 vector sets)")
    logger.info("=" * 60)

    bar_kernels = build_bar_kernels().to(device)

    # Pre-compute inverse matrices for all vector sets
    inv_matrices = {}
    for name, vecs in VECTOR_SETS.items():
        inv_matrices[name] = build_stain_inverse(vecs["hematoxylin"], vecs["dab"], vecs["residual"])
        logger.info("  Vector set '%s': H=%s, DAB=%s", name,
                     [round(v, 3) for v in vecs["hematoxylin"]],
                     [round(v, 3) for v in vecs["dab"]])

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use NVMe tiles if available
    nvme_tiles = Path("/opt/training_data/tiles")
    tile_base = nvme_tiles if nvme_tiles.exists() else data_dir / "tiles"
    mask_base = data_dir / "masks"

    tile_dirs = sorted([d for d in tile_base.iterdir() if d.is_dir()])
    total = 0
    skipped = 0

    for slide_dir in tqdm(tile_dirs, desc="Multi-vector labels"):
        slide_name = slide_dir.name
        out_dir = output_dir / slide_name
        out_dir.mkdir(parents=True, exist_ok=True)

        tile_paths = sorted(slide_dir.glob("*.tiff")) + sorted(slide_dir.glob("*.png"))

        for tile_path in tile_paths:
            out_path = out_dir / f"{tile_path.stem}_soft.npy"
            if out_path.exists():
                skipped += 1
                continue

            # Load tile
            if tile_path.suffix == ".tiff":
                img = tifffile.imread(str(tile_path))
            else:
                img = cv2.cvtColor(cv2.imread(str(tile_path)), cv2.COLOR_BGR2RGB)
            if img is None:
                continue

            H, W = img.shape[:2]

            # Load masks
            nuc_path = mask_base / slide_name / f"{tile_path.stem}_nuclei.tiff"
            cell_path = mask_base / slide_name / f"{tile_path.stem}_cells.tiff"
            if not nuc_path.exists() or not cell_path.exists():
                continue

            nuc_mask = tifffile.imread(str(nuc_path)) > 0
            cell_mask = tifffile.imread(str(cell_path)) > 0

            # Accumulate membrane probability across vector sets
            membrane_accum = np.zeros((H, W), dtype=np.float32)

            for name, inv_mat in inv_matrices.items():
                # Extract DAB with this vector set
                dab = extract_dab_with_vectors(img, inv_mat)

                # Bar-filter on GPU
                dab_t = torch.from_numpy(dab).float().to(device).unsqueeze(0).unsqueeze(0)
                pad = bar_kernels.shape[-1] // 2
                responses = F.conv2d(dab_t, bar_kernels, padding=pad)
                bar_max = responses.max(dim=1).values.squeeze().cpu().numpy()

                # Membrane for this vector set: bar-filter positive within cell, outside nucleus
                bar_positive = bar_max > 0
                membrane_this = bar_positive & cell_mask & ~nuc_mask
                membrane_accum += membrane_this.astype(np.float32)

            # Average membrane probability: how many of 7 vector sets agree
            n_sets = len(VECTOR_SETS)
            membrane_prob = membrane_accum / n_sets  # 0 to 1

            # Build soft 4-class label
            soft_label = np.zeros((4, H, W), dtype=np.float32)

            # Class 0: background (not in any cell)
            bg = ~cell_mask
            soft_label[0, bg] = 1.0

            # Class 1: nucleus
            soft_label[1, nuc_mask] = 1.0

            # Class 3: membrane (averaged probability)
            cell_non_nuc = cell_mask & ~nuc_mask
            soft_label[3, cell_non_nuc] = membrane_prob[cell_non_nuc]

            # Class 2: cytoplasm (cell interior that's not nucleus or membrane)
            soft_label[2, cell_non_nuc] = 1.0 - membrane_prob[cell_non_nuc]

            # Normalize each pixel to sum to 1
            pixel_sum = soft_label.sum(axis=0, keepdims=True)
            pixel_sum[pixel_sum == 0] = 1
            soft_label = soft_label / pixel_sum

            np.save(str(out_path), soft_label.astype(np.float16))
            total += 1

    logger.info("Stage 1 complete: %d labels generated, %d skipped (existing)", total, skipped)
    return total


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: Train Gen-3 (reuse megazord training infrastructure)
# ═══════════════════════════════════════════════════════════════════════════

def stage2_train_gen3(data_dir: Path, label_dir: Path, save_path: Path, device: str):
    """Train Gen-3 on multi-vector averaged labels."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Training Gen-3 on multi-vector labels")
    logger.info("=" * 60)

    # Import training infrastructure from megazord
    sys.path.insert(0, str(PROJECT / "scripts"))
    from importlib import import_module

    # We need the model, dataset, loss, and training loop from the megazord script
    # Rather than importing (complex), redefine the key components here

    # Model classes (same as Gen-2)
    from scripts_gen3_model import WiderAttnUNet, SoftLabelDataset, SoftCEDiceLoss, train_model

    # Use NVMe tiles if available
    nvme_tiles = Path("/opt/training_data/tiles")
    tile_base = nvme_tiles if nvme_tiles.exists() else data_dir / "tiles"

    # Collect tiles with multi-vector labels
    tiles = []
    for slide_dir in sorted(tile_base.iterdir()):
        if not slide_dir.is_dir():
            continue
        for tp in sorted(slide_dir.glob("*.tiff")) + sorted(slide_dir.glob("*.png")):
            soft_path = label_dir / slide_dir.name / f"{tp.stem}_soft.npy"
            if soft_path.exists():
                tiles.append(tp)

    logger.info("Found %d tiles with multi-vector labels", len(tiles))

    # Slide-level split
    slides = sorted(set(p.parent.name for p in tiles))
    rng = np.random.default_rng(SEED)
    rng.shuffle(slides)
    nt, nv = int(len(slides) * 0.7), int(len(slides) * 0.15)
    tr_s, va_s = set(slides[:nt]), set(slides[nt:nt + nv])
    train_tiles = [p for p in tiles if p.parent.name in tr_s]
    val_tiles = [p for p in tiles if p.parent.name in va_s]
    logger.info("Split: %d train / %d val tiles", len(train_tiles), len(val_tiles))

    train_ds = SoftLabelDataset(train_tiles, label_dir, augment=True)
    val_ds = SoftLabelDataset(val_tiles, label_dir, augment=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True,
                              persistent_workers=True, prefetch_factor=8)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=8, pin_memory=True,
                            persistent_workers=True, prefetch_factor=8)

    # Initialize Gen-3 from Gen-2 weights
    model = WiderAttnUNet(channels=(48, 96, 192, 384)).to(device)
    gen2_path = PROJECT / "models" / "membrane_gen2_wider_attn_best.pth"
    if gen2_path.exists():
        ckpt = torch.load(gen2_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Initialized from Gen-2 (epoch %d, val_loss=%.4f)", ckpt["epoch"], ckpt["val_loss"])
    else:
        logger.warning("Gen-2 checkpoint not found — training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("WiderAttnUNet: %.2fM params (batch=64, multi-vector labels)", n_params / 1e6)

    loss_fn = SoftCEDiceLoss()
    best_ep, best_vl = train_model(
        model, train_loader, val_loader, loss_fn, device,
        epochs=999, lr=5e-4, patience=20, save_path=save_path, soft_labels=True,
    )
    logger.info("Stage 2 complete: best epoch %d, val_loss=%.4f", best_ep, best_vl)
    return best_ep, best_vl


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = "cuda"

    cfg = load_config(str(PROJECT / "config" / "default.yaml"))
    data_dir = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    eval_dir = Path(cfg["paths"]["evaluation_dir"])

    multivector_label_dir = Path("/opt/training_data/membrane_multivector_labels")
    gen3_path = model_dir / "membrane_gen3_multivector_best.pth"

    # Stage 1: Generate multi-vector labels
    n_labels = stage1_generate_multivector_labels(data_dir, multivector_label_dir, device)

    # Stage 2: Train Gen-3
    # For now, just import the training components from megazord
    # We'll write a standalone version if the import is too complex
    logger.info("=" * 60)
    logger.info("Stage 2: Import training infrastructure...")
    logger.info("=" * 60)

    # Import model and training classes from megazord
    exec(open(str(PROJECT / "scripts" / "19_megazord_pipeline.py")).read(),
         {"__name__": "megazord_import"})

    # Actually, let's just use the megazord's stage2 function directly
    # by pointing it to the multi-vector labels
    from pathlib import Path as _P

    # The megazord's stage2_train_gen2 function works perfectly —
    # it just needs different label_dir and save_path
    logger.info("Launching Gen-3 training via megazord stage2 infrastructure...")

    # We need to run stage2_train_gen2 with our multi-vector labels
    # The simplest way: modify the megazord call

    # Use megazord's full training loop
    import importlib.util
    spec = importlib.util.spec_from_file_location("megazord",
        str(PROJECT / "scripts" / "19_megazord_pipeline.py"))
    megazord = importlib.util.module_from_spec(spec)

    # Override the label dir in the function
    # Actually this is getting too complex. Let me just inline the training.

    logger.info("Training Gen-3 inline...")

    # ── Inline training (same as megazord stage2 but with different labels) ──

    nvme_tiles = Path("/opt/training_data/tiles")
    tile_base = nvme_tiles if nvme_tiles.exists() else data_dir / "tiles"

    tiles = []
    for slide_dir in sorted(tile_base.iterdir()):
        if not slide_dir.is_dir():
            continue
        for tp in sorted(slide_dir.glob("*.tiff")) + sorted(slide_dir.glob("*.png")):
            soft_path = multivector_label_dir / slide_dir.name / f"{tp.stem}_soft.npy"
            if soft_path.exists():
                tiles.append(tp)

    slides = sorted(set(p.parent.name for p in tiles))
    rng = np.random.default_rng(SEED)
    rng.shuffle(slides)
    nt, nv = int(len(slides) * 0.7), int(len(slides) * 0.15)
    tr_s, va_s = set(slides[:nt]), set(slides[nt:nt + nv])
    train_tiles = [p for p in tiles if p.parent.name in tr_s]
    val_tiles = [p for p in tiles if p.parent.name in va_s]
    logger.info("Split: %d train / %d val tiles", len(train_tiles), len(val_tiles))

    # Dataset defined at module level (SoftDS) for spawn pickling compatibility

    train_ds = SoftDS(train_tiles, multivector_label_dir, augment=True)
    val_ds = SoftDS(val_tiles, multivector_label_dir, augment=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True,
                              persistent_workers=True, prefetch_factor=8)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=8, pin_memory=True,
                            persistent_workers=True, prefetch_factor=8)

    # Model and loss classes defined at module level for spawn compatibility
    model = WiderAttnUNet().to(device)
    gen2_path = model_dir / "membrane_gen2_wider_attn_best.pth"
    if gen2_path.exists():
        ckpt = torch.load(gen2_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Initialized Gen-3 from Gen-2 (epoch %d, vl=%.4f)", ckpt["epoch"], ckpt["val_loss"])

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("WiderAttnUNet: %.2fM params", n_params / 1e6)

    # ── Training loop (same as megazord with full checkpoint) ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-7)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = SoftCEDiceLoss()
    best_vl, best_ep, epochs_no_improve = float("inf"), 0, 0

    # Resume if checkpoint exists with optimizer state
    if gen3_path.exists():
        ckpt = torch.load(gen3_path, weights_only=False, map_location=device)
        if "optimizer_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_vl = ckpt.get("val_loss", float("inf"))
            best_ep = ckpt.get("best_epoch", ckpt.get("epoch", 0))
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            start_epoch = ckpt.get("epoch", 0) + 1
            logger.info("Resumed Gen-3 from epoch %d (vl=%.4f)", start_epoch - 1, best_vl)
        else:
            start_epoch = 1
    else:
        start_epoch = 1

    for ep in range(start_epoch, 1000):
        t_ep = time.time()
        model.train()
        train_loss, n = 0.0, 0
        for rgb, lbl in train_loader:
            rgb, lbl = rgb.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = loss_fn(model(rgb), lbl)
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
                rgb, lbl = rgb.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    loss = loss_fn(model(rgb), lbl)
                val_loss += loss.item()
                vn += 1
                pred = model(rgb).argmax(dim=1) if not torch.amp.is_autocast_enabled() else lbl.argmax(dim=1)
                # Quick accuracy from last batch
                tgt = lbl.argmax(dim=1)
                for c in range(4):
                    m = tgt == c
                    cc[c] += (model(rgb).argmax(dim=1)[m] == c).sum() if False else (pred[m] == c).sum()
                    ct[c] += m.sum()

        vl = val_loss / max(vn, 1)
        scheduler.step(vl)
        acc = (cc / ct.clamp(min=1)).cpu().numpy()
        lr = optimizer.param_groups[0]["lr"]

        logger.info("Ep %3d | t=%.4f v=%.4f | bg=%.2f nu=%.2f cy=%.2f me=%.2f | lr=%.1e | %.0fs",
                     ep, train_loss / max(n, 1), vl, acc[0], acc[1], acc[2], acc[3], lr, time.time() - t_ep)

        if vl < best_vl:
            best_vl, best_ep = vl, ep
            epochs_no_improve = 0
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": ep, "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": vl, "per_class_acc": acc.tolist(),
                "best_epoch": best_ep, "epochs_no_improve": epochs_no_improve,
                "vector_sets": list(VECTOR_SETS.keys()),
            }, gen3_path)
            logger.info("  -> Best (ep %d, vl=%.4f)", ep, vl)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 20:
                logger.info("Early stopping at ep %d (best: ep %d, vl=%.4f)", ep, best_ep, best_vl)
                break

    # ── Gap validation ──
    logger.info("=" * 60)
    logger.info("Gen-3 Gap Validation")
    logger.info("=" * 60)

    # Load best checkpoint
    ckpt = torch.load(gen3_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sc = cfg["stain_deconvolution"]
    from instanseg_brightfield.stain import StainDeconvolver
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))

    ds = torch.load(data_dir / "segmentation_dataset_base.pth", weights_only=False)
    test_data = ds["Test"][:200]

    argmax_mem, argmax_cyto, soft_mem, soft_cyto = [], [], [], []

    for item in tqdm(test_data, desc="Gap validation"):
        img = tifffile.imread(str(data_dir / item["image"]))
        if img is None:
            continue
        dab = deconv.extract_dab(img)
        rgb_t = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.amp.autocast("cuda"):
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

    results = {"generation": "Gen-3", "vector_sets": list(VECTOR_SETS.keys())}
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

    logger.info("REF: Gen-2 argmax=+0.209, Gen-1 argmax=+0.208, bar-filter=+0.127")

    json.dump(results, open(eval_dir / "gen3_multivector_gap_results.json", "w"), indent=2)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("GEN-3 COMPLETE in %.1f hours", elapsed / 3600)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
