"""Branch H: Real Virchow2 epithelial filter + DAB classification.

Filters the base dataset to keep only:
1. Tiles classified as epithelial via Virchow2 embedding clustering
2. Cells with DAB class >= 1+ (mean DAB OD >= 0.10)

Model loading strategy (tries in order):
  1. Real Virchow2 via timm (requires HuggingFace gated access)
  2. Fallback: pretrained ResNet50 feature extraction + KMeans clustering

In both paths the pipeline:
  - Extracts tile-level embeddings
  - KMeans(n_clusters=2) to separate epithelial vs stromal
  - Validates cluster identity: epithelial cluster has higher mean DAB
  - Keeps only tiles assigned to the epithelial cluster
  - Removes DAB-negative cells (mean DAB OD < 0.10)

Requires:
- data/segmentation_dataset_base.pth (from script 03)
- Tile images + masks already generated

Produces: data/segmentation_dataset_branch_h.pth
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAB classification
# ---------------------------------------------------------------------------

DAB_THRESHOLDS = {
    "negative": 0.10,   # below this = negative
    "1+": 0.20,         # 0.10 - 0.20
    "2+": 0.35,         # 0.20 - 0.35
    "3+": float("inf"), # >= 0.35
}


def _classify_dab(dab_mean: float) -> str:
    """Map mean DAB optical density to IHC intensity class."""
    if dab_mean < DAB_THRESHOLDS["negative"]:
        return "Negative"
    elif dab_mean < DAB_THRESHOLDS["1+"]:
        return "1+"
    elif dab_mean < DAB_THRESHOLDS["2+"]:
        return "2+"
    else:
        return "3+"


# ---------------------------------------------------------------------------
# Virchow2 model loading (real model with gated HF access)
# ---------------------------------------------------------------------------

def _load_virchow2_model(device: str = "cpu"):
    """Load the real Virchow2 foundation model via timm.

    Requires HuggingFace gated access to ``paige-ai/Virchow2``.

    Returns:
        model: Virchow2 model in eval mode.
        transform: Preprocessing transform for input images.
    """
    import timm

    logger.info("Loading Virchow2 model from HuggingFace hub (paige-ai/Virchow2)...")

    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model = model.to(device)
    model.eval()

    transform_config = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**transform_config, is_training=False)

    logger.info("Virchow2 loaded on device: %s", device)
    return model, transform


def _extract_virchow2_embeddings(
    model: torch.nn.Module,
    transform: Any,
    tile_images: list[np.ndarray],
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Extract Virchow2 embeddings: concat(CLS, mean(patch_tokens)) = 2560-d.

    Args:
        model: Virchow2 model in eval mode.
        transform: torchvision/timm preprocessing transform.
        tile_images: List of RGB uint8 numpy arrays (H, W, 3).
        device: Torch device string.
        batch_size: Inference batch size.

    Returns:
        L2-normalized numpy array of shape (n_tiles, 2560), dtype float32.
    """
    from PIL import Image

    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(tile_images), batch_size):
        batch_imgs = tile_images[start:start + batch_size]
        tensors = []
        for img_np in batch_imgs:
            pil_img = Image.fromarray(img_np)
            tensors.append(transform(pil_img))
        batch = torch.stack(tensors)

        is_cuda = str(device).startswith("cuda")
        if is_cuda:
            batch = batch.pin_memory().to(device, non_blocking=True)
        else:
            batch = batch.to(device)

        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=is_cuda,
        ):
            output = model(batch)

        # Virchow2 returns (B, n_tokens, dim).  Token 0 = CLS.
        if hasattr(output, "shape") and output.ndim == 3:
            cls_token = output[:, 0, :]
            patch_tokens = output[:, 1:, :].mean(dim=1)
            embeddings = torch.cat([cls_token, patch_tokens], dim=-1)
        else:
            embeddings = output

        embeddings = embeddings.cpu().numpy().astype(np.float32)
        all_embeddings.append(embeddings)

    result = np.concatenate(all_embeddings, axis=0)

    # L2-normalize
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    result = result / norms
    return result


# ---------------------------------------------------------------------------
# Fallback: pretrained ResNet50 feature extractor
# ---------------------------------------------------------------------------

def _load_resnet50_extractor(device: str = "cpu"):
    """Load a pretrained torchvision ResNet50 as a feature extractor.

    The final classification head is removed so that forward() returns
    the 2048-d average-pooled feature vector.

    Returns:
        model: ResNet50 feature extractor in eval mode.
        transform: Standard ImageNet preprocessing transform.
    """
    import torchvision.models as models
    import torchvision.transforms as T

    logger.info("Loading pretrained ResNet50 as fallback feature extractor...")

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    resnet = models.resnet50(weights=weights)

    # Remove the final fc layer so forward() returns the pooled features.
    # We replace it with Identity to get the 2048-d vector.
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    logger.info("ResNet50 feature extractor loaded on device: %s", device)
    return resnet, transform


def _extract_resnet50_features(
    model: torch.nn.Module,
    transform: Any,
    tile_images: list[np.ndarray],
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Extract 2048-d ResNet50 features from tile images.

    Args:
        model: ResNet50 with Identity fc head, in eval mode.
        transform: ImageNet preprocessing transform.
        tile_images: List of RGB uint8 numpy arrays (H, W, 3).
        device: Torch device string.
        batch_size: Inference batch size.

    Returns:
        L2-normalized numpy array of shape (n_tiles, 2048), dtype float32.
    """
    from PIL import Image

    all_features: list[np.ndarray] = []

    for start in range(0, len(tile_images), batch_size):
        batch_imgs = tile_images[start:start + batch_size]
        tensors = []
        for img_np in batch_imgs:
            pil_img = Image.fromarray(img_np)
            tensors.append(transform(pil_img))
        batch = torch.stack(tensors)

        is_cuda = str(device).startswith("cuda")
        if is_cuda:
            batch = batch.pin_memory().to(device, non_blocking=True)
        else:
            batch = batch.to(device)

        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=is_cuda,
        ):
            features = model(batch)

        features = features.cpu().numpy().astype(np.float32)
        all_features.append(features)

    result = np.concatenate(all_features, axis=0)

    # L2-normalize
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    result = result / norms
    return result


# ---------------------------------------------------------------------------
# Tile embedding extraction (auto-selects Virchow2 or ResNet50 fallback)
# ---------------------------------------------------------------------------

def _load_tile_images(
    data_dir: Path,
    items: list[dict],
) -> list[np.ndarray]:
    """Load RGB tile images for the given dataset items.

    Args:
        data_dir: Root data directory.
        items: List of dataset item dicts with ``"image"`` keys.

    Returns:
        List of RGB uint8 numpy arrays (H, W, 3).
    """
    images: list[np.ndarray] = []
    for item in items:
        img = tifffile.imread(data_dir / item["image"])
        images.append(img)
    return images


def _compute_embeddings_for_items(
    data_dir: Path,
    items: list[dict],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, str]:
    """Compute tile embeddings using Virchow2 (or ResNet50 fallback).

    Returns:
        embeddings: numpy array (n_tiles, embed_dim).
        method: ``"virchow2"`` or ``"resnet50_fallback"``.
    """
    tile_images = _load_tile_images(data_dir, items)

    # Try Virchow2 first
    try:
        model, transform = _load_virchow2_model(device)
        embeddings = _extract_virchow2_embeddings(
            model, transform, tile_images, device=device, batch_size=batch_size,
        )
        del model
        torch.cuda.empty_cache() if device.startswith("cuda") else None
        return embeddings, "virchow2"
    except Exception as exc:
        logger.warning(
            "Virchow2 not available (%s). Falling back to ResNet50 features.", exc
        )

    # Fallback: pretrained ResNet50
    model, transform = _load_resnet50_extractor(device)
    embeddings = _extract_resnet50_features(
        model, transform, tile_images, device=device, batch_size=batch_size,
    )
    del model
    torch.cuda.empty_cache() if device.startswith("cuda") else None
    return embeddings, "resnet50_fallback"


# ---------------------------------------------------------------------------
# Epithelial cluster identification via KMeans + DAB validation
# ---------------------------------------------------------------------------

def _identify_epithelial_cluster(
    embeddings: np.ndarray,
    dab_means: list[float],
) -> tuple[np.ndarray, int]:
    """Cluster tile embeddings and identify the epithelial cluster.

    Uses KMeans(n_clusters=2) and validates by checking which cluster has
    higher mean DAB intensity (epithelial tissue in CLDN18.2 IHC staining
    has more DAB-positive cells than stroma).

    Args:
        embeddings: L2-normalized array (n_tiles, embed_dim).
        dab_means: Per-tile mean DAB optical density, parallel to embeddings.

    Returns:
        cluster_labels: Integer array of cluster assignments (0 or 1).
        epithelial_cluster: Which cluster ID is epithelial.
    """
    logger.info("Running KMeans(n_clusters=2) on %d tile embeddings (dim=%d)...",
                len(embeddings), embeddings.shape[1])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(embeddings)

    cluster_sizes = [int((cluster_labels == i).sum()) for i in range(2)]
    logger.info("  Cluster 0: %d tiles", cluster_sizes[0])
    logger.info("  Cluster 1: %d tiles", cluster_sizes[1])

    # Validate via DAB intensity
    dab_arr = np.array(dab_means)
    mean_dab_0 = float(dab_arr[cluster_labels == 0].mean()) if cluster_sizes[0] > 0 else 0.0
    mean_dab_1 = float(dab_arr[cluster_labels == 1].mean()) if cluster_sizes[1] > 0 else 0.0

    logger.info("  Mean DAB (cluster 0): %.4f", mean_dab_0)
    logger.info("  Mean DAB (cluster 1): %.4f", mean_dab_1)

    if mean_dab_0 > mean_dab_1:
        epithelial_cluster = 0
    elif mean_dab_1 > mean_dab_0:
        epithelial_cluster = 1
    else:
        # Tie-breaker: epithelial tissue is typically denser (smaller cluster
        # in gastric tissue where stroma is more abundant)
        epithelial_cluster = 0 if cluster_sizes[0] <= cluster_sizes[1] else 1

    logger.info("  -> Cluster %d = Epithelial (higher DAB)", epithelial_cluster)
    logger.info("  -> Cluster %d = Stromal (lower DAB)", 1 - epithelial_cluster)

    return cluster_labels, epithelial_cluster


# ---------------------------------------------------------------------------
# Main filter pipeline
# ---------------------------------------------------------------------------

def filter_dataset_branch_h(
    cfg: dict,
    output_path: Path | None = None,
    device: str = "auto",
    batch_size: int = 32,
) -> Path:
    """Apply Branch H filtering: Virchow2 epithelial clustering + DAB class.

    Args:
        cfg: Pipeline configuration dictionary.
        output_path: Where to save filtered dataset. Defaults to
            ``data/segmentation_dataset_branch_h.pth``.
        device: Torch device (``"auto"``, ``"cuda"``, or ``"cpu"``).
        batch_size: Batch size for embedding extraction.

    Returns:
        Path to the saved filtered dataset.
    """
    data_dir = Path(cfg["paths"]["data_dir"])
    sc = cfg["stain_deconvolution"]
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))

    input_path = data_dir / "segmentation_dataset_base.pth"
    if output_path is None:
        output_path = data_dir / "segmentation_dataset_branch_h.pth"

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = torch.load(input_path, weights_only=False)

    # Minimum DAB OD to keep a cell
    DAB_NEGATIVE_THRESHOLD = 0.10

    filtered: dict[str, list[dict]] = {"Train": [], "Validation": [], "Test": []}
    stats = {
        "total": 0,
        "kept": 0,
        "dropped_stromal": 0,
        "dropped_negative_dab": 0,
        "dropped_no_cells": 0,
    }

    for split in ["Train", "Validation", "Test"]:
        items = dataset[split]
        if not items:
            logger.info("Skipping empty split: %s", split)
            continue

        logger.info("Processing %s: %d items", split, len(items))

        # Step 1: Compute per-tile DAB means (needed for cluster validation)
        logger.info("  Computing per-tile DAB means...")
        tile_dab_means: list[float] = []
        valid_indices: list[int] = []

        for idx, item in enumerate(tqdm(items, desc=f"DAB {split}")):
            img = tifffile.imread(data_dir / item["image"])
            nuc = tifffile.imread(data_dir / item["nucleus_masks"]).astype(np.int32)
            if nuc.max() == 0:
                tile_dab_means.append(0.0)
                continue
            dab = deconv.extract_dab(img)
            tile_dab_means.append(float(dab.mean()))
            valid_indices.append(idx)

        # Step 2: Compute embeddings for all tiles (including empty ones, for
        # consistent indexing; empty tiles will simply get low scores)
        logger.info("  Extracting tile embeddings...")
        embeddings, method = _compute_embeddings_for_items(
            data_dir, items, device=device, batch_size=batch_size,
        )
        logger.info("  Embedding method: %s, shape: %s", method, embeddings.shape)

        # Step 3: Cluster and identify epithelial cluster
        cluster_labels, epithelial_cluster = _identify_epithelial_cluster(
            embeddings, tile_dab_means,
        )

        # Step 4: Filter tiles and cells
        logger.info("  Filtering tiles and cells...")
        for idx, item in enumerate(tqdm(items, desc=f"Branch H {split}")):
            stats["total"] += 1

            nuc = tifffile.imread(data_dir / item["nucleus_masks"]).astype(np.int32)
            cell = tifffile.imread(data_dir / item["cell_masks"]).astype(np.int32)

            if nuc.max() == 0:
                stats["dropped_no_cells"] += 1
                continue

            # Filter 1: Must be in the epithelial cluster
            if cluster_labels[idx] != epithelial_cluster:
                stats["dropped_stromal"] += 1
                continue

            # Filter 2: Remove DAB-negative cells
            img = tifffile.imread(data_dir / item["image"])
            dab = deconv.extract_dab(img)

            cell_ids = np.unique(cell)
            cell_ids = cell_ids[cell_ids > 0]

            keep_ids: set[int] = set()
            for cid in cell_ids:
                cell_mask = cell == cid
                dab_mean = float(dab[cell_mask].mean())
                if dab_mean >= DAB_NEGATIVE_THRESHOLD:
                    keep_ids.add(int(cid))

            if not keep_ids:
                stats["dropped_negative_dab"] += 1
                continue

            # Rebuild masks with only kept cells
            keep_list = list(keep_ids)
            nuc_filtered = np.where(np.isin(nuc, keep_list), nuc, 0)
            cell_filtered = np.where(np.isin(cell, keep_list), cell, 0)

            # Save filtered masks back
            nuc_path = data_dir / item["nucleus_masks"]
            cell_path = data_dir / item["cell_masks"]
            tifffile.imwrite(str(nuc_path), nuc_filtered.astype(np.uint16), compression="zlib")
            tifffile.imwrite(str(cell_path), cell_filtered.astype(np.uint16), compression="zlib")

            filtered[split].append(item)
            stats["kept"] += 1

    # Subsample to ~10k for RAM safety (max 50 tiles per slide)
    rng = random.Random(42)
    for split in filtered:
        items = filtered[split]
        slides: dict[str, list[dict]] = {}
        for item in items:
            slide = Path(item["image"]).parts[1]
            slides.setdefault(slide, []).append(item)
        sampled: list[dict] = []
        for _slide_name, slide_items in sorted(slides.items()):
            sampled.extend(rng.sample(slide_items, min(50, len(slide_items))))
        filtered[split] = sampled

    total_filtered = sum(len(v) for v in filtered.values())
    logger.info("Branch H filtering complete:")
    logger.info("  Total: %d -> %d (%.1f%% kept)", stats["total"], stats["kept"],
                stats["kept"] / max(stats["total"], 1) * 100)
    logger.info("  Dropped stromal: %d", stats["dropped_stromal"])
    logger.info("  Dropped negative DAB: %d", stats["dropped_negative_dab"])
    logger.info("  Dropped no cells: %d", stats["dropped_no_cells"])
    logger.info("  After subsampling: %d items", total_filtered)

    for split in filtered:
        logger.info("  %s: %d", split, len(filtered[split]))

    torch.save(filtered, output_path)
    logger.info("Saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Branch H: Virchow2 epithelial filter + DAB classification.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: 'auto', 'cuda', or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction (default: 32).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output .pth path.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    cfg = load_config(args.config)
    output_path = Path(args.output) if args.output else None
    filter_dataset_branch_h(cfg, output_path=output_path, device=args.device,
                            batch_size=args.batch_size)


if __name__ == "__main__":
    main()
