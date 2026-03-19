#!/usr/bin/env python3
"""Score tiles with multiple foundation pathology models for training data filtering.

Runs every available foundation model on pre-extracted tiles and produces a
single CSV (data/tile_scores.csv) with per-tile scores from all models.
Models that fail to load are gracefully skipped.

Models (in order of attempted loading):
    1. Virchow2       (2560-dim, KMeans epithelial clustering)
    2. CONCH 1.5      (512-dim, zero-shot text similarity)
    3. UNI2-H         (1536-dim, KMeans epithelial clustering)
    4. TITAN           (768-dim, zero-shot text similarity)
    5. ResNet50        (2048-dim, KMeans epithelial clustering, always available)

Each tile also receives DAB-mean and cell-density statistics.  A consensus
score (mean of all available model scores) is appended for downstream
filtering.

Usage:
    python scripts/score_tiles_foundation_models.py --config config/default.yaml
    python scripts/score_tiles_foundation_models.py --device cuda:0 --batch-size 128
    python scripts/score_tiles_foundation_models.py --max-tiles 500  # quick test
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver, build_stain_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_INPUT_SIZE = 224  # All foundation models expect 224x224
TILE_SOURCE_SIZE = 512  # Our tiles are 512x512 on disk


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Container for a single model's scoring output."""
    name: str
    csv_column: str
    scores: np.ndarray          # (n_tiles,) float32, 0-1
    embedding_dim: int
    device_used: str
    time_seconds: float


@dataclass
class TileRecord:
    """Metadata for a single tile."""
    tile_path: str
    slide_name: str
    dab_mean: float = 0.0
    cell_density: float = 0.0


# ---------------------------------------------------------------------------
# Tile discovery and loading
# ---------------------------------------------------------------------------

def discover_tiles(data_dir: Path) -> list[TileRecord]:
    """Find all PNG tiles under data/tiles/<slide_name>/*.png.

    Returns a list of TileRecord with tile_path and slide_name populated.
    """
    tile_dir = data_dir / "tiles"
    if not tile_dir.exists():
        logger.error("Tile directory not found: %s", tile_dir)
        return []

    records = []
    for slide_dir in sorted(tile_dir.iterdir()):
        if not slide_dir.is_dir():
            continue
        slide_name = slide_dir.name
        for tile_path in sorted(slide_dir.glob("*.png")):
            records.append(TileRecord(
                tile_path=str(tile_path),
                slide_name=slide_name,
            ))

    return records


def load_tile_images(
    records: list[TileRecord],
    target_size: int = MODEL_INPUT_SIZE,
) -> list[Image.Image]:
    """Load tiles from disk, resizing to target_size x target_size.

    Returns PIL Images in RGB mode, matching the order of *records*.
    """
    images = []
    for rec in tqdm(records, desc="Loading tiles", leave=False):
        img = Image.open(rec.tile_path).convert("RGB")
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.LANCZOS)
        images.append(img)
    return images


# ---------------------------------------------------------------------------
# DAB and cell density computation
# ---------------------------------------------------------------------------

def compute_dab_stats(
    records: list[TileRecord],
    stain_cfg: dict,
) -> None:
    """Compute mean DAB intensity for each tile (modifies records in-place)."""
    stain_matrix = build_stain_matrix(
        stain_cfg["hematoxylin"],
        stain_cfg["dab"],
        stain_cfg["residual"],
    )
    deconvolver = StainDeconvolver(stain_matrix)

    for rec in tqdm(records, desc="Computing DAB stats", leave=False):
        img = np.array(Image.open(rec.tile_path).convert("RGB"))
        dab = deconvolver.extract_dab(img)
        rec.dab_mean = float(np.mean(dab))


def compute_cell_density(
    records: list[TileRecord],
    data_dir: Path,
    tile_size_px: int = TILE_SOURCE_SIZE,
    target_mpp: float = 0.5,
) -> None:
    """Estimate cell density from existing mask files (modifies records in-place).

    Looks for corresponding mask .tiff files under data/masks/<slide>/<tile>.tiff.
    Cell density = number of unique labels / tile area in mm^2.
    If masks are not yet generated, leaves cell_density at 0.
    """
    masks_dir = data_dir / "masks"
    if not masks_dir.exists():
        logger.info("Masks directory not found (%s), cell_density will be 0.", masks_dir)
        return

    tile_area_mm2 = (tile_size_px * target_mpp / 1000.0) ** 2
    if tile_area_mm2 <= 0:
        return

    for rec in records:
        tile_p = Path(rec.tile_path)
        # Try several mask naming conventions
        mask_candidates = [
            masks_dir / rec.slide_name / (tile_p.stem + ".tiff"),
            masks_dir / rec.slide_name / (tile_p.stem + "_cells.tiff"),
            masks_dir / rec.slide_name / (tile_p.stem + "_mask.tiff"),
            masks_dir / rec.slide_name / (tile_p.stem + ".png"),
        ]
        for mask_path in mask_candidates:
            if mask_path.exists():
                try:
                    import tifffile
                    mask = tifffile.imread(str(mask_path))
                except Exception:
                    mask = np.array(Image.open(mask_path))
                n_cells = len(np.unique(mask)) - 1  # subtract background
                n_cells = max(n_cells, 0)
                rec.cell_density = float(n_cells / tile_area_mm2)
                break


# ---------------------------------------------------------------------------
# Model availability probes and loaders
# ---------------------------------------------------------------------------

def _select_devices(preferred: str) -> list[str]:
    """Return a list of usable CUDA devices, or ['cpu'].

    When *preferred* is 'auto', uses all visible GPUs.  Otherwise returns a
    single-element list with the explicitly requested device.
    """
    import torch

    if preferred != "auto":
        return [preferred]

    if not torch.cuda.is_available():
        return ["cpu"]

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return ["cpu"]

    return [f"cuda:{i}" for i in range(n_gpus)]


def _assign_device(devices: list[str], model_index: int) -> str:
    """Round-robin device assignment for multi-GPU setups."""
    return devices[model_index % len(devices)]


# -- Virchow2 ----------------------------------------------------------------

def _try_load_virchow2(device: str):
    """Attempt to load Virchow2 via timm.  Returns (model, transform) or None."""
    try:
        import timm
        import torch

        logger.info("  Loading Virchow2 via timm (hf-hub:paige-ai/Virchow2)...")
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        model = model.to(device).eval()

        transform_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**transform_cfg, is_training=False)

        logger.info("  Virchow2 loaded on %s", device)
        return model, transform
    except Exception as exc:
        logger.warning("  Virchow2 unavailable: %s", exc)
        return None


def _extract_virchow2(model, transform, images, device, batch_size):
    """Extract 2560-dim Virchow2 embeddings (CLS + mean_patch)."""
    import torch

    all_emb = []
    is_cuda = str(device).startswith("cuda")

    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start + batch_size]
        tensors = torch.stack([transform(img) for img in batch_imgs])
        if is_cuda:
            tensors = tensors.pin_memory().to(device, non_blocking=True)
        else:
            tensors = tensors.to(device)

        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=is_cuda,
        ):
            output = model(tensors)
            if hasattr(output, "shape") and output.ndim == 3:
                cls_tok = output[:, 0, :]
                patch_mean = output[:, 1:, :].mean(dim=1)
                emb = torch.cat([cls_tok, patch_mean], dim=1)
            else:
                emb = output

        all_emb.append(emb.cpu().numpy().astype(np.float32))

    return np.concatenate(all_emb, axis=0)


# -- CONCH 1.5 ----------------------------------------------------------------

def _try_load_conch(device: str):
    """Load CONCH (v1 or v1.5) with tokenizer for zero-shot.

    Returns (model, transform, tokenizer, loading_method) or None.
    """
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Strategy 1: conch library
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        import torch

        logger.info("  Loading CONCH via conch library...")
        model, transform = create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
            hf_auth_token=hf_token,
        )
        model = model.to(device).eval()

        tokenizer = None
        try:
            from conch.open_clip_custom import get_tokenizer
            tokenizer = get_tokenizer()
        except ImportError:
            pass

        logger.info("  CONCH loaded via conch library on %s", device)
        return model, transform, tokenizer, "conch"
    except Exception as exc:
        logger.debug("  conch library not available: %s", exc)

    # Strategy 2: timm (vision-only, no zero-shot)
    try:
        import timm
        import torch

        logger.info("  Loading CONCH via timm...")
        model = timm.create_model("hf-hub:MahmoodLab/CONCH", pretrained=True)
        model = model.to(device).eval()

        transform_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**transform_cfg, is_training=False)

        logger.info("  CONCH loaded via timm on %s (no tokenizer)", device)
        return model, transform, None, "timm"
    except Exception as exc:
        logger.debug("  timm loading failed: %s", exc)

    # Strategy 3: open_clip
    try:
        import open_clip
        import torch

        logger.info("  Loading CONCH via open_clip...")
        model, _, transform = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="hf-hub:MahmoodLab/conch",
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        logger.info("  CONCH loaded via open_clip on %s", device)
        return model, transform, tokenizer, "open_clip"
    except Exception as exc:
        logger.debug("  open_clip loading failed: %s", exc)

    logger.warning("  CONCH unavailable via all loading strategies.")
    return None


def _run_conch_forward(model, batch, loading_method):
    """Single forward pass returning raw embedding tensor."""
    if loading_method == "conch" and hasattr(model, "encode_image"):
        return model.encode_image(batch, proj_contrast=False, normalize=False)
    elif loading_method == "open_clip" and hasattr(model, "encode_image"):
        return model.encode_image(batch, normalize=False)
    else:
        out = model(batch)
        if hasattr(out, "shape") and out.ndim == 3:
            out = out[:, 0, :]
        return out


def _extract_conch(model, transform, images, device, batch_size, loading_method):
    """Extract L2-normalised CONCH embeddings."""
    import torch

    all_emb = []
    is_cuda = str(device).startswith("cuda")

    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start + batch_size]
        tensors = torch.stack([transform(img) for img in batch_imgs])
        if is_cuda:
            tensors = tensors.pin_memory().to(device, non_blocking=True)
        else:
            tensors = tensors.to(device)

        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=is_cuda,
        ):
            emb = _run_conch_forward(model, tensors, loading_method)

        all_emb.append(emb.cpu().numpy().astype(np.float32))

    result = np.concatenate(all_emb, axis=0)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return result / norms


def _conch_zero_shot_score(
    model,
    tokenizer,
    embeddings: np.ndarray,
    device: str,
) -> np.ndarray | None:
    """Compute 0-1 score via cosine similarity to positive/negative prompts.

    Positive prompt: "strong membranous staining"
    Negative prompt: "negative membranous staining"

    Returns per-tile score in [0, 1] (higher = more positive staining).
    """
    import torch

    if tokenizer is None or not hasattr(model, "encode_text"):
        return None

    prompts = {
        "negative": "negative membranous staining",
        "positive": "strong membranous staining",
    }

    text_embs = {}
    for key, prompt in prompts.items():
        try:
            tokenized = tokenizer([prompt], return_tensors="pt")
            tokens = tokenized["input_ids"].to(device)
        except (TypeError, KeyError):
            tokens = tokenizer([prompt]).to(device)
        with torch.inference_mode():
            t_emb = model.encode_text(tokens)
            t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
            text_embs[key] = t_emb.cpu().numpy().astype(np.float32)

    # Cosine similarities (embeddings already L2-normed)
    sim_neg = (embeddings @ text_embs["negative"].T).squeeze(-1)
    sim_pos = (embeddings @ text_embs["positive"].T).squeeze(-1)

    # Softmax-like normalization to [0, 1]
    stacked = np.stack([sim_neg, sim_pos], axis=1)
    exp_s = np.exp(stacked - stacked.max(axis=1, keepdims=True))
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    return probs[:, 1].astype(np.float32)


# -- UNI2-H ------------------------------------------------------------------

def _try_load_uni2h(device: str):
    """Attempt to load UNI2-H via timm.  Returns (model, transform) or None."""
    try:
        import timm
        import torch

        logger.info("  Loading UNI2-H via timm (hf-hub:MahmoodLab/uni2-h)...")
        model = timm.create_model(
            "hf-hub:MahmoodLab/uni2-h",
            pretrained=True,
        )
        model = model.to(device).eval()

        transform_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**transform_cfg, is_training=False)

        logger.info("  UNI2-H loaded on %s", device)
        return model, transform
    except Exception as exc:
        logger.warning("  UNI2-H unavailable: %s", exc)
        return None


def _extract_uni2h(model, transform, images, device, batch_size):
    """Extract 1536-dim UNI2-H embeddings."""
    import torch

    all_emb = []
    is_cuda = str(device).startswith("cuda")

    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start + batch_size]
        tensors = torch.stack([transform(img) for img in batch_imgs])
        if is_cuda:
            tensors = tensors.pin_memory().to(device, non_blocking=True)
        else:
            tensors = tensors.to(device)

        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=is_cuda,
        ):
            output = model(tensors)
            if hasattr(output, "shape") and output.ndim == 3:
                emb = output[:, 0, :]  # CLS token
            else:
                emb = output

        all_emb.append(emb.cpu().numpy().astype(np.float32))

    return np.concatenate(all_emb, axis=0)


# -- TITAN --------------------------------------------------------------------

def _try_load_titan(device: str):
    """Load TITAN model with tokenizer for zero-shot.

    Returns (model, transform, tokenizer, loading_method) or None.
    """
    # Strategy 1: titan library
    try:
        from titan.open_clip_custom import create_model_from_pretrained
        import torch

        logger.info("  Loading TITAN via titan library...")
        model, transform = create_model_from_pretrained(
            "titan_ViT-L-14", "hf_hub:MahmoodLab/titan",
        )
        model = model.to(device).eval()

        tokenizer = None
        try:
            from titan.open_clip_custom import get_tokenizer
            tokenizer = get_tokenizer()
        except ImportError:
            pass

        logger.info("  TITAN loaded via titan library on %s", device)
        return model, transform, tokenizer, "titan"
    except Exception as exc:
        logger.debug("  titan library not available: %s", exc)

    # Strategy 2: timm
    try:
        import timm
        import torch

        logger.info("  Loading TITAN via timm...")
        model = timm.create_model("hf-hub:MahmoodLab/titan", pretrained=True)
        model = model.to(device).eval()

        transform_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**transform_cfg, is_training=False)

        logger.info("  TITAN loaded via timm on %s (no tokenizer)", device)
        return model, transform, None, "timm"
    except Exception as exc:
        logger.debug("  timm loading failed: %s", exc)

    # Strategy 3: open_clip
    try:
        import open_clip
        import torch

        logger.info("  Loading TITAN via open_clip...")
        model, _, transform = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="hf-hub:MahmoodLab/titan",
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        logger.info("  TITAN loaded via open_clip on %s", device)
        return model, transform, tokenizer, "open_clip"
    except Exception as exc:
        logger.debug("  open_clip loading failed: %s", exc)

    logger.warning("  TITAN unavailable via all loading strategies.")
    return None


def _run_titan_forward(model, batch, loading_method):
    """Single forward pass returning raw embedding tensor."""
    if loading_method == "titan" and hasattr(model, "encode_image"):
        return model.encode_image(batch, proj_contrast=False, normalize=False)
    elif loading_method == "open_clip" and hasattr(model, "encode_image"):
        return model.encode_image(batch, normalize=False)
    else:
        out = model(batch)
        if hasattr(out, "shape") and out.ndim == 3:
            out = out[:, 0, :]
        return out


def _extract_titan(model, transform, images, device, batch_size, loading_method):
    """Extract L2-normalised TITAN embeddings."""
    import torch

    all_emb = []
    is_cuda = str(device).startswith("cuda")

    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start + batch_size]
        tensors = torch.stack([transform(img) for img in batch_imgs])
        if is_cuda:
            tensors = tensors.pin_memory().to(device, non_blocking=True)
        else:
            tensors = tensors.to(device)

        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=is_cuda,
        ):
            emb = _run_titan_forward(model, tensors, loading_method)

        all_emb.append(emb.cpu().numpy().astype(np.float32))

    result = np.concatenate(all_emb, axis=0)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return result / norms


def _titan_zero_shot_score(
    model,
    tokenizer,
    embeddings: np.ndarray,
    device: str,
) -> np.ndarray | None:
    """Compute 0-1 score via cosine similarity (same prompts as CONCH)."""
    import torch

    if tokenizer is None or not hasattr(model, "encode_text"):
        return None

    prompts = {
        "negative": "negative membranous staining",
        "positive": "strong membranous staining",
    }

    text_embs = {}
    for key, prompt in prompts.items():
        try:
            tokenized = tokenizer([prompt], return_tensors="pt")
            tokens = tokenized["input_ids"].to(device)
        except (TypeError, KeyError):
            tokens = tokenizer([prompt]).to(device)
        with torch.inference_mode():
            t_emb = model.encode_text(tokens)
            t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
            text_embs[key] = t_emb.cpu().numpy().astype(np.float32)

    sim_neg = (embeddings @ text_embs["negative"].T).squeeze(-1)
    sim_pos = (embeddings @ text_embs["positive"].T).squeeze(-1)

    stacked = np.stack([sim_neg, sim_pos], axis=1)
    exp_s = np.exp(stacked - stacked.max(axis=1, keepdims=True))
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    return probs[:, 1].astype(np.float32)


# -- ResNet50 (always available) ----------------------------------------------

def _load_resnet50(device: str):
    """Load torchvision ResNet50 pretrained on ImageNet.  Always succeeds."""
    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    logger.info("  Loading ResNet50 (ImageNet V2)...")
    model = models.resnet50(weights="IMAGENET1K_V2")
    # Remove the classification head to get 2048-dim features
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    transform = T.Compose([
        T.Resize(MODEL_INPUT_SIZE),
        T.CenterCrop(MODEL_INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    logger.info("  ResNet50 loaded on %s", device)
    return model, transform


def _extract_resnet50(model, transform, images, device, batch_size):
    """Extract 2048-dim ResNet50 features."""
    import torch

    all_emb = []
    is_cuda = str(device).startswith("cuda")

    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start + batch_size]
        tensors = torch.stack([transform(img) for img in batch_imgs])
        if is_cuda:
            tensors = tensors.pin_memory().to(device, non_blocking=True)
        else:
            tensors = tensors.to(device)

        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=is_cuda,
        ):
            emb = model(tensors)

        all_emb.append(emb.cpu().numpy().astype(np.float32))

    return np.concatenate(all_emb, axis=0)


# ---------------------------------------------------------------------------
# KMeans scoring (for embedding-based models)
# ---------------------------------------------------------------------------

def kmeans_epithelial_score(
    embeddings: np.ndarray,
    dab_values: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """Run KMeans(2) on embeddings and identify the epithelial cluster via DAB.

    The cluster whose tiles have higher mean DAB intensity is deemed epithelial.
    Returns per-tile scores in [0, 1] (soft assignment distance to epithelial
    centroid, normalized).

    Falls back to a size heuristic (smaller cluster = epithelial) when DAB
    data is unavailable.
    """
    from sklearn.cluster import KMeans

    n_tiles = len(embeddings)
    if n_tiles < 4:
        logger.warning("  %s: too few tiles (%d) for KMeans, returning 0.5", model_name, n_tiles)
        return np.full(n_tiles, 0.5, dtype=np.float32)

    # L2-normalise before clustering
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X = (embeddings / norms).astype(np.float32)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X)

    cluster_sizes = [int((labels == i).sum()) for i in range(2)]
    logger.info("  %s KMeans: cluster 0 = %d tiles, cluster 1 = %d tiles",
                model_name, cluster_sizes[0], cluster_sizes[1])

    # Identify epithelial cluster via DAB intensity
    dab_per_cluster = {0: [], 1: []}
    valid_dab = np.isfinite(dab_values) & (dab_values > 0)
    for i in range(n_tiles):
        if valid_dab[i]:
            dab_per_cluster[labels[i]].append(dab_values[i])

    if dab_per_cluster[0] and dab_per_cluster[1]:
        mean_dab_0 = float(np.mean(dab_per_cluster[0]))
        mean_dab_1 = float(np.mean(dab_per_cluster[1]))
        epithelial_cluster = 0 if mean_dab_0 > mean_dab_1 else 1
        logger.info("  %s: mean DAB cluster 0=%.4f (%d tiles), cluster 1=%.4f (%d tiles)",
                    model_name, mean_dab_0, len(dab_per_cluster[0]),
                    mean_dab_1, len(dab_per_cluster[1]))
        logger.info("  %s: cluster %d = Epithelial (higher DAB)", model_name, epithelial_cluster)
    else:
        # Fallback: smaller cluster = epithelial (stroma is more abundant)
        epithelial_cluster = 0 if cluster_sizes[0] < cluster_sizes[1] else 1
        logger.info("  %s: no DAB data, using size heuristic: cluster %d = Epithelial (smaller)",
                    model_name, epithelial_cluster)

    # Compute soft scores: distance-based probability of belonging to
    # the epithelial cluster.  Use inverse of distance to each centroid,
    # normalised to [0, 1].
    c_epi = kmeans.cluster_centers_[epithelial_cluster]
    c_str = kmeans.cluster_centers_[1 - epithelial_cluster]

    dist_epi = np.linalg.norm(X - c_epi[np.newaxis, :], axis=1)
    dist_str = np.linalg.norm(X - c_str[np.newaxis, :], axis=1)

    # Avoid division by zero
    total = dist_epi + dist_str + 1e-8
    scores = (dist_str / total).astype(np.float32)  # closer to epi -> higher score

    return scores


# ---------------------------------------------------------------------------
# Unload helpers
# ---------------------------------------------------------------------------

def _unload_model(*objects) -> None:
    """Delete model objects and free GPU memory."""
    import torch

    for obj in objects:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main scoring pipeline
# ---------------------------------------------------------------------------

def run_scoring(
    records: list[TileRecord],
    images: list[Image.Image],
    cfg: dict,
    device_pref: str = "auto",
    batch_size: int = 64,
) -> list[ModelResult]:
    """Run all available models and return a list of ModelResult."""
    import torch

    devices = _select_devices(device_pref)
    logger.info("Available devices: %s", devices)

    dab_values = np.array([r.dab_mean for r in records], dtype=np.float32)
    results: list[ModelResult] = []
    model_idx = 0

    # ---- 1. Virchow2 --------------------------------------------------------
    dev = _assign_device(devices, model_idx)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Model 1/5: Virchow2 (device=%s)", dev)
    logger.info("=" * 60)

    loaded = _try_load_virchow2(dev)
    if loaded is not None:
        model, transform = loaded
        t0 = time.time()
        embeddings = _extract_virchow2(model, transform, images, dev, batch_size)
        scores = kmeans_epithelial_score(embeddings, dab_values, "Virchow2")
        elapsed = time.time() - t0
        results.append(ModelResult(
            name="Virchow2", csv_column="virchow2_score", scores=scores,
            embedding_dim=embeddings.shape[1], device_used=dev,
            time_seconds=elapsed,
        ))
        _unload_model(model, transform, embeddings)
        logger.info("  Virchow2 done in %.1fs", elapsed)
    else:
        logger.info("  Virchow2 SKIPPED")
    model_idx += 1

    # ---- 2. CONCH 1.5 --------------------------------------------------------
    dev = _assign_device(devices, model_idx)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Model 2/5: CONCH 1.5 (device=%s)", dev)
    logger.info("=" * 60)

    loaded = _try_load_conch(dev)
    if loaded is not None:
        model, transform, tokenizer, method = loaded
        t0 = time.time()
        embeddings = _extract_conch(model, transform, images, dev, batch_size, method)

        # Prefer zero-shot scoring if tokenizer is available
        scores = _conch_zero_shot_score(model, tokenizer, embeddings, dev)
        if scores is None:
            logger.info("  CONCH: no tokenizer, falling back to KMeans scoring")
            scores = kmeans_epithelial_score(embeddings, dab_values, "CONCH")

        elapsed = time.time() - t0
        results.append(ModelResult(
            name="CONCH", csv_column="conch_score", scores=scores,
            embedding_dim=embeddings.shape[1], device_used=dev,
            time_seconds=elapsed,
        ))
        _unload_model(model, transform, tokenizer, embeddings)
        logger.info("  CONCH done in %.1fs", elapsed)
    else:
        logger.info("  CONCH SKIPPED")
    model_idx += 1

    # ---- 3. UNI2-H -----------------------------------------------------------
    dev = _assign_device(devices, model_idx)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Model 3/5: UNI2-H (device=%s)", dev)
    logger.info("=" * 60)

    loaded = _try_load_uni2h(dev)
    if loaded is not None:
        model, transform = loaded
        t0 = time.time()
        embeddings = _extract_uni2h(model, transform, images, dev, batch_size)
        scores = kmeans_epithelial_score(embeddings, dab_values, "UNI2-H")
        elapsed = time.time() - t0
        results.append(ModelResult(
            name="UNI2-H", csv_column="uni2h_score", scores=scores,
            embedding_dim=embeddings.shape[1], device_used=dev,
            time_seconds=elapsed,
        ))
        _unload_model(model, transform, embeddings)
        logger.info("  UNI2-H done in %.1fs", elapsed)
    else:
        logger.info("  UNI2-H SKIPPED")
    model_idx += 1

    # ---- 4. TITAN ------------------------------------------------------------
    dev = _assign_device(devices, model_idx)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Model 4/5: TITAN (device=%s)", dev)
    logger.info("=" * 60)

    loaded = _try_load_titan(dev)
    if loaded is not None:
        model, transform, tokenizer, method = loaded
        t0 = time.time()
        embeddings = _extract_titan(model, transform, images, dev, batch_size, method)

        scores = _titan_zero_shot_score(model, tokenizer, embeddings, dev)
        if scores is None:
            logger.info("  TITAN: no tokenizer, falling back to KMeans scoring")
            scores = kmeans_epithelial_score(embeddings, dab_values, "TITAN")

        elapsed = time.time() - t0
        results.append(ModelResult(
            name="TITAN", csv_column="titan_score", scores=scores,
            embedding_dim=embeddings.shape[1], device_used=dev,
            time_seconds=elapsed,
        ))
        _unload_model(model, transform, tokenizer, embeddings)
        logger.info("  TITAN done in %.1fs", elapsed)
    else:
        logger.info("  TITAN SKIPPED")
    model_idx += 1

    # ---- 5. ResNet50 (always available) --------------------------------------
    dev = _assign_device(devices, model_idx)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Model 5/5: ResNet50 (device=%s)", dev)
    logger.info("=" * 60)

    model, transform = _load_resnet50(dev)
    t0 = time.time()
    embeddings = _extract_resnet50(model, transform, images, dev, batch_size)
    scores = kmeans_epithelial_score(embeddings, dab_values, "ResNet50")
    elapsed = time.time() - t0
    results.append(ModelResult(
        name="ResNet50", csv_column="resnet50_score", scores=scores,
        embedding_dim=embeddings.shape[1], device_used=dev,
        time_seconds=elapsed,
    ))
    _unload_model(model, transform, embeddings)
    logger.info("  ResNet50 done in %.1fs", elapsed)

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

# Canonical column order for the output CSV
ALL_SCORE_COLUMNS = [
    "virchow2_score",
    "conch_score",
    "uni2h_score",
    "titan_score",
    "resnet50_score",
]


def write_scores_csv(
    records: list[TileRecord],
    results: list[ModelResult],
    output_path: Path,
) -> None:
    """Write tile_scores.csv with all model scores and metadata.

    Writes atomically via tempfile + os.replace.
    """
    n_tiles = len(records)

    # Build a lookup from csv_column -> scores array
    score_lookup: dict[str, np.ndarray] = {}
    for r in results:
        score_lookup[r.csv_column] = r.scores

    fieldnames = ["tile_path", "slide_name"]
    fieldnames.extend(ALL_SCORE_COLUMNS)
    fieldnames.extend(["dab_mean", "cell_density", "consensus_score"])

    rows = []
    for i, rec in enumerate(records):
        row: dict[str, Any] = {
            "tile_path": rec.tile_path,
            "slide_name": rec.slide_name,
        }

        # Individual model scores (empty string if model was skipped)
        available_scores = []
        for col in ALL_SCORE_COLUMNS:
            if col in score_lookup:
                val = float(score_lookup[col][i])
                row[col] = f"{val:.6f}"
                available_scores.append(val)
            else:
                row[col] = ""

        row["dab_mean"] = f"{rec.dab_mean:.6f}"
        row["cell_density"] = f"{rec.cell_density:.2f}"

        # Consensus = mean of all available model scores
        if available_scores:
            row["consensus_score"] = f"{float(np.mean(available_scores)):.6f}"
        else:
            row["consensus_score"] = ""

        rows.append(row)

    # Atomic write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(output_path.parent), suffix=".tmp", prefix=output_path.stem,
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, str(output_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Wrote %d tile scores to %s", n_tiles, output_path)


# ---------------------------------------------------------------------------
# Summary logging
# ---------------------------------------------------------------------------

def log_summary(
    records: list[TileRecord],
    results: list[ModelResult],
) -> None:
    """Print summary statistics."""
    n_tiles = len(records)

    logger.info("")
    logger.info("=" * 60)
    logger.info("SCORING SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total tiles:  %d", n_tiles)
    logger.info("  Total slides: %d", len({r.slide_name for r in records}))
    logger.info("")

    # Per-model stats
    logger.info("  %-12s  %-6s  %-8s  %-8s  %-8s  %-6s  %s",
                "Model", "Dim", "Mean", "Median", "Std", "Device", "Time")
    logger.info("  " + "-" * 70)

    available_names = []
    for r in results:
        s = r.scores
        available_names.append(r.name)
        logger.info("  %-12s  %4d    %.4f    %.4f    %.4f   %-6s  %.1fs",
                    r.name, r.embedding_dim,
                    float(np.mean(s)), float(np.median(s)), float(np.std(s)),
                    r.device_used, r.time_seconds)

    # Models that were skipped
    all_names = {"Virchow2", "CONCH", "UNI2-H", "TITAN", "ResNet50"}
    skipped = all_names - set(available_names)
    if skipped:
        logger.info("")
        logger.info("  Skipped models: %s", ", ".join(sorted(skipped)))

    # DAB stats
    dab_vals = np.array([r.dab_mean for r in records])
    logger.info("")
    logger.info("  DAB mean:   mean=%.4f  median=%.4f  std=%.4f",
                float(np.mean(dab_vals)), float(np.median(dab_vals)),
                float(np.std(dab_vals)))

    # Consensus stats
    if results:
        all_scores = np.stack([r.scores for r in results], axis=1)
        consensus = np.mean(all_scores, axis=1)
        logger.info("  Consensus:  mean=%.4f  median=%.4f  std=%.4f",
                    float(np.mean(consensus)), float(np.median(consensus)),
                    float(np.std(consensus)))

    total_time = sum(r.time_seconds for r in results)
    logger.info("")
    logger.info("  Total scoring time: %.1fs (%.2f tiles/sec across all models)",
                total_time, n_tiles * len(results) / max(total_time, 1e-3))
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score tiles with foundation pathology models for data filtering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config (default: config/default.yaml).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1' (default: auto).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Inference batch size per model (default: 64).",
    )
    parser.add_argument(
        "--max-tiles", type=int, default=0,
        help="Limit total tiles processed (0 = no limit, for testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    stain_cfg = cfg["stain_deconvolution"]
    target_mpp = cfg["tile_extraction"]["target_mpp"]
    tile_size = cfg["tile_extraction"]["tile_size"]

    output_path = data_dir / "tile_scores.csv"

    logger.info("Foundation Model Tile Scoring Pipeline")
    logger.info("=" * 60)
    logger.info("  Config:     %s", args.config or "config/default.yaml")
    logger.info("  Data dir:   %s", data_dir)
    logger.info("  Output:     %s", output_path)
    logger.info("  Device:     %s", args.device)
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("  Max tiles:  %s", args.max_tiles if args.max_tiles > 0 else "all")
    logger.info("")

    # ------------------------------------------------------------------
    # Step 1: Discover tiles
    # ------------------------------------------------------------------
    logger.info("Step 1: Discovering tiles...")
    records = discover_tiles(data_dir)
    if not records:
        logger.error("No tiles found under %s/tiles/", data_dir)
        return

    if args.max_tiles > 0 and len(records) > args.max_tiles:
        import random
        rng = random.Random(42)
        records = rng.sample(records, args.max_tiles)
        logger.info("  Subsampled to %d tiles (--max-tiles)", args.max_tiles)

    logger.info("  Found %d tiles from %d slides",
                len(records), len({r.slide_name for r in records}))
    logger.info("")

    # ------------------------------------------------------------------
    # Step 2: Compute DAB and cell density
    # ------------------------------------------------------------------
    logger.info("Step 2: Computing DAB intensity and cell density...")
    compute_dab_stats(records, stain_cfg)
    compute_cell_density(records, data_dir, tile_size_px=tile_size, target_mpp=target_mpp)
    logger.info("  DAB: mean=%.4f, cell density: mean=%.2f cells/mm^2",
                float(np.mean([r.dab_mean for r in records])),
                float(np.mean([r.cell_density for r in records])))
    logger.info("")

    # ------------------------------------------------------------------
    # Step 3: Load tile images (resized to 224x224)
    # ------------------------------------------------------------------
    logger.info("Step 3: Loading tile images (resized to %dx%d)...",
                MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    images = load_tile_images(records, target_size=MODEL_INPUT_SIZE)
    logger.info("  Loaded %d images", len(images))
    logger.info("")

    # ------------------------------------------------------------------
    # Step 4: Score with all available models
    # ------------------------------------------------------------------
    logger.info("Step 4: Scoring with foundation models...")
    results = run_scoring(
        records, images, cfg,
        device_pref=args.device,
        batch_size=args.batch_size,
    )

    # Free images after scoring
    del images

    # ------------------------------------------------------------------
    # Step 5: Write CSV
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Step 5: Writing scores to %s", output_path)
    write_scores_csv(records, results, output_path)

    # ------------------------------------------------------------------
    # Step 6: Summary
    # ------------------------------------------------------------------
    log_summary(records, results)


if __name__ == "__main__":
    main()
