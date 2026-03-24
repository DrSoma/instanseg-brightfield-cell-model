#!/usr/bin/env python3
"""Coloring-book heatmap: pixel-level membrane overlay for one slide.

Uses the Gen-1 membrane U-Net to classify every pixel as:
  - Background → transparent
  - Nucleus → dark blue fill
  - Cytoplasm → light fill colored by cell's mean DAB
  - Membrane → crisp line colored by local DAB intensity (blue→red)

Produces a high-res RGBA overlay PNG that looks like a coloring book
with membrane outlines and colored interiors.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/20_coloring_book_heatmap.py --slide CLDN0042
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import openslide
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stderr, force=True)
logger = logging.getLogger("heatmap")

# ── Model (same as training script) ──
class _DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True),
                                   nn.Conv2d(o,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True))
    def forward(self, x): return self.block(x)
class _Down(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(i, o))
    def forward(self, x): return self.pool_conv(x)
class _Up(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.up = nn.ConvTranspose2d(i, i//2, 2, stride=2); self.conv = _DoubleConv(i, o)
    def forward(self, x, sk):
        x = self.up(x); dy, dx = sk.size(2)-x.size(2), sk.size(3)-x.size(3)
        x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2]); return self.conv(torch.cat([sk, x], 1))
class MembraneUNet4C(nn.Module):
    def __init__(self, ch=(32,64,128,256)):
        super().__init__()
        c0,c1,c2,c3 = ch
        self.inc = _DoubleConv(3, c0); self.down1 = _Down(c0, c1); self.down2 = _Down(c1, c2)
        self.down3 = _Down(c2, c3); self.bottleneck = _Down(c3, c3*2)
        self.up1 = _Up(c3*2, c3); self.up2 = _Up(c3, c2); self.up3 = _Up(c2, c1); self.up4 = _Up(c1, c0)
        self.outc = nn.Conv2d(c0, 4, 1)
    def forward(self, x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.bottleneck(x4)
        return self.outc(self.up4(self.up3(self.up2(self.up1(x5,x4),x3),x2),x1))

# ── Colormap for DAB intensity ──
# Blue (low DAB) → Yellow → Red (high DAB)
DAB_CMAP = plt.cm.RdYlBu_r

# ── Colors ──
NUC_COLOR = np.array([30, 50, 120, 200], dtype=np.uint8)   # Dark blue, semi-opaque
BG_COLOR = np.array([0, 0, 0, 0], dtype=np.uint8)           # Transparent


def generate_heatmap(slide_name: str, model, deconv, device: str, output_dir: Path,
                     tile_size: int = 512, target_mpp: float = 0.5, output_width: int = 8192):
    """Generate a coloring-book heatmap for one slide."""
    t0 = time.time()

    # Find slide
    slide_dirs = [Path("/media/fernandosoto/DATA/CLDN18 slides"), Path("/tmp/bc_slides")]
    slide_path = None
    for sd in slide_dirs:
        for ext in [".ndpi", ".svs", ".tiff"]:
            p = sd / f"{slide_name}{ext}"
            if p.exists():
                slide_path = p
                break
        if slide_path:
            break

    if slide_path is None:
        logger.error("Slide %s not found", slide_name)
        return

    logger.info("Slide: %s", slide_path)
    wsi = openslide.OpenSlide(str(slide_path))

    # Find read level close to target MPP
    mpp_x = float(wsi.properties.get("openslide.mpp-x", 0.25))
    best_level = 0
    for level in range(wsi.level_count):
        ds = wsi.level_downsamples[level]
        level_mpp = mpp_x * ds
        if level_mpp <= target_mpp * 1.1:
            best_level = level
    ds = wsi.level_downsamples[best_level]
    level_dims = wsi.level_dimensions[best_level]
    logger.info("Read level %d: %dx%d (ds=%.1f, mpp=%.3f)", best_level,
                level_dims[0], level_dims[1], ds, mpp_x * ds)

    # Output dimensions
    scale = output_width / level_dims[0]
    out_h = int(level_dims[1] * scale)
    logger.info("Output: %dx%d", output_width, out_h)

    # Create output RGBA canvas
    canvas = np.zeros((out_h, output_width, 4), dtype=np.uint8)

    # Tile grid
    step = tile_size
    n_cols = (level_dims[0] + step - 1) // step
    n_rows = (level_dims[1] + step - 1) // step
    total_tiles = n_cols * n_rows
    logger.info("Tile grid: %dx%d = %d tiles", n_cols, n_rows, total_tiles)

    processed = 0
    for row in tqdm(range(n_rows), desc="Rows"):
        for col in range(n_cols):
            # Read tile at level coordinates
            x_level = col * step
            y_level = row * step
            x_l0 = int(x_level * ds)
            y_l0 = int(y_level * ds)

            # Read from WSI
            w = min(tile_size, level_dims[0] - x_level)
            h = min(tile_size, level_dims[1] - y_level)
            if w < 32 or h < 32:
                continue

            region = wsi.read_region((x_l0, y_l0), best_level, (w, h))
            tile_rgb = np.array(region.convert("RGB"))

            # Skip mostly white tiles
            if tile_rgb.mean() > 235:
                continue

            # Pad to tile_size if needed
            if w < tile_size or h < tile_size:
                padded = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
                padded[:h, :w] = tile_rgb
                tile_rgb = padded

            # DAB extraction
            dab = deconv.extract_dab(tile_rgb)

            # Model prediction
            rgb_t = torch.from_numpy(tile_rgb.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.inference_mode(), torch.amp.autocast("cuda"):
                logits = model(rgb_t)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()  # (4, H, W)
            pred = probs.argmax(axis=0)  # (H, W)

            # Build RGBA tile overlay
            tile_overlay = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)

            # Nucleus pixels → dark blue
            nuc_mask = pred == 1
            tile_overlay[nuc_mask] = NUC_COLOR

            # Cytoplasm pixels → light green/teal fill, alpha proportional to cell probability
            cyto_mask = pred == 2
            cyto_prob = probs[2]
            tile_overlay[cyto_mask, 0] = 100  # R
            tile_overlay[cyto_mask, 1] = 180  # G
            tile_overlay[cyto_mask, 2] = 140  # B
            tile_overlay[cyto_mask, 3] = (cyto_prob[cyto_mask] * 120).clip(0, 255).astype(np.uint8)

            # Membrane pixels → colored by DAB intensity, full opacity
            mem_mask = pred == 3
            if mem_mask.any():
                # Normalize DAB to [0, 1] for colormap
                dab_norm = np.clip(dab / 0.6, 0, 1)  # 0.6 OD = very strong
                colors = (DAB_CMAP(dab_norm) * 255).astype(np.uint8)  # (H, W, 4)
                tile_overlay[mem_mask, 0] = colors[mem_mask, 0]
                tile_overlay[mem_mask, 1] = colors[mem_mask, 1]
                tile_overlay[mem_mask, 2] = colors[mem_mask, 2]
                # Alpha: membrane probability × 230 (near-opaque)
                mem_prob = probs[3]
                tile_overlay[mem_mask, 3] = (mem_prob[mem_mask] * 230).clip(0, 255).astype(np.uint8)

            # Crop to actual tile size (remove padding)
            tile_overlay = tile_overlay[:h, :w]

            # Scale and paste into canvas
            out_x = int(x_level * scale)
            out_y = int(y_level * scale)
            out_w = int(w * scale)
            out_h_tile = int(h * scale)

            if out_w > 0 and out_h_tile > 0:
                resized = cv2.resize(tile_overlay, (out_w, out_h_tile), interpolation=cv2.INTER_AREA)
                # Paste with alpha compositing
                y1, y2 = out_y, min(out_y + out_h_tile, canvas.shape[0])
                x1, x2 = out_x, min(out_x + out_w, canvas.shape[1])
                rh, rw = y2 - y1, x2 - x1
                if rh > 0 and rw > 0:
                    canvas[y1:y2, x1:x2] = resized[:rh, :rw]

            processed += 1

    wsi.close()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{slide_name}_coloring_book.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGRA))

    elapsed = time.time() - t0
    logger.info("Saved: %s (%dx%d, %d tiles, %.1f min)",
                out_path, canvas.shape[1], canvas.shape[0], processed, elapsed / 60)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", type=str, default="CLDN0042", help="Slide name (without extension)")
    parser.add_argument("--output-width", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config("config/default.yaml")
    sc = cfg["stain_deconvolution"]
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))

    # Load model
    model = MembraneUNet4C().to(args.device)
    ckpt = torch.load("models/membrane_unet4c_best.pth", weights_only=False, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded (epoch %d)", ckpt["epoch"])

    output_dir = Path("evaluation/coloring_book_heatmaps")
    generate_heatmap(args.slide, model, deconv, args.device, output_dir,
                     output_width=args.output_width)


if __name__ == "__main__":
    main()
