"""Branch I: CONCH-style zero-shot tile filter + DAB classification.

Scores each tile using features that approximate what CONCH's zero-shot
text prompts describe for CLDN18.2 membranous staining:
  - "negative CLDN18.2 membranous staining"  -> low DAB, no ring patterns
  - "strong CLDN18.2 membranous staining"    -> high DAB, clear ring patterns

Model loading strategy (tries in order):
  1. Real CONCH model via conch / timm / open_clip
  2. Fallback: pure-vision scoring based on DAB intensity, membrane ring
     pattern, cell density, and color saturation

Filtering steps:
  1. For each tile, compute a 0-1 staining presence score
  2. Keep tiles scoring above threshold
  3. Remove DAB-negative cells (mean DAB OD < 0.10)

Requires:
- data/segmentation_dataset_base.pth (from script 03)
- Tile images + masks already generated

Produces: data/segmentation_dataset_branch_i.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch
from tqdm import tqdm

from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONCH zero-shot prompts (same as 10b_conch_pipeline.py)
# ---------------------------------------------------------------------------

ZERO_SHOT_PROMPTS = {
    "absent_staining": (
        "negative CLDN18.2 membranous staining, no brown DAB chromogen on cell membranes"
    ),
    "weak_staining": (
        "weak CLDN18.2 membranous staining, faint brown DAB chromogen on cell membranes"
    ),
    "moderate_staining": (
        "moderate CLDN18.2 membranous staining, distinct brown DAB chromogen on cell membranes"
    ),
    "strong_staining": (
        "strong CLDN18.2 membranous staining, intense dark brown DAB chromogen on cell membranes"
    ),
}


# ---------------------------------------------------------------------------
# DAB classification
# ---------------------------------------------------------------------------

DAB_NEGATIVE_THRESHOLD = 0.10  # cells below this are negative


# ---------------------------------------------------------------------------
# CONCH model loading (real model, multiple strategies)
# ---------------------------------------------------------------------------

def _load_conch_model(device: str = "cpu"):
    """Load CONCH model, trying multiple strategies in order.

    Strategies:
        1. conch library (create_model_from_pretrained)
        2. timm (via HuggingFace hub)
        3. open_clip

    Returns:
        model: CONCH model in eval mode.
        transform: Preprocessing transform.
        tokenizer: Text tokenizer (for zero-shot), or None.
        method: String indicating which loading method succeeded.

    Raises:
        RuntimeError: If no loading strategy succeeds.
    """
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Strategy 1: conch library
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        logger.info("Loading CONCH via conch library...")
        model, transform = create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
            hf_auth_token=hf_token,
        )
        model = model.to(device)
        model.eval()

        tokenizer = None
        try:
            from conch.open_clip_custom import get_tokenizer
            tokenizer = get_tokenizer()
        except ImportError:
            pass

        logger.info("CONCH loaded via conch library on device: %s", device)
        return model, transform, tokenizer, "conch"
    except (ImportError, Exception) as exc:
        logger.debug("conch library not available: %s", exc)

    # Strategy 2: timm
    try:
        import timm
        logger.info("Loading CONCH via timm...")
        model = timm.create_model("hf-hub:MahmoodLab/CONCH", pretrained=True)
        model = model.to(device)
        model.eval()

        transform_config = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**transform_config, is_training=False)

        logger.info("CONCH loaded via timm on device: %s", device)
        return model, transform, None, "timm"
    except (ImportError, Exception) as exc:
        logger.debug("timm loading failed: %s", exc)

    # Strategy 3: open_clip
    try:
        import open_clip
        logger.info("Loading CONCH via open_clip...")
        model, _, transform = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="hf-hub:MahmoodLab/conch",
        )
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        logger.info("CONCH loaded via open_clip on device: %s", device)
        return model, transform, tokenizer, "open_clip"
    except (ImportError, Exception) as exc:
        logger.debug("open_clip loading failed: %s", exc)

    raise RuntimeError("Could not load CONCH model via any method (conch/timm/open_clip).")


def _encode_text_prompts(
    model: Any,
    tokenizer: Any,
    prompts: dict[str, str],
    device: str = "cpu",
) -> dict[str, np.ndarray] | None:
    """Encode zero-shot text prompts into L2-normalized embeddings.

    Args:
        model: CONCH model with ``encode_text`` method.
        tokenizer: Text tokenizer.
        prompts: Mapping from class name to prompt string.
        device: Torch device string.

    Returns:
        Dict mapping class name to numpy embedding (1, embed_dim), or None
        if the model does not support text encoding.
    """
    if tokenizer is None or not hasattr(model, "encode_text"):
        return None

    text_embeddings: dict[str, np.ndarray] = {}
    for class_name, prompt in prompts.items():
        try:
            tokenized = tokenizer([prompt], return_tensors="pt")
            tokens = tokenized["input_ids"].to(device)
        except (TypeError, KeyError):
            tokens = tokenizer([prompt]).to(device)
        with torch.inference_mode():
            text_emb = model.encode_text(tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            text_embeddings[class_name] = text_emb.cpu().numpy().astype(np.float32)

    return text_embeddings


def _extract_conch_embeddings(
    model: Any,
    transform: Any,
    tile_images: list[np.ndarray],
    device: str = "cpu",
    batch_size: int = 64,
    loading_method: str = "conch",
) -> np.ndarray:
    """Extract L2-normalized CONCH embeddings from tile images.

    Args:
        model: CONCH model in eval mode.
        transform: Preprocessing transform.
        tile_images: List of RGB uint8 numpy arrays (H, W, 3).
        device: Torch device string.
        batch_size: Inference batch size.
        loading_method: Which loading strategy was used.

    Returns:
        L2-normalized numpy array (n_tiles, 512), dtype float32.
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
            if loading_method == "conch" and hasattr(model, "encode_image"):
                emb = model.encode_image(batch, proj_contrast=False, normalize=False)
            elif loading_method == "open_clip" and hasattr(model, "encode_image"):
                emb = model.encode_image(batch, normalize=False)
            else:
                emb = model(batch)
                if hasattr(emb, "shape") and emb.ndim == 3:
                    emb = emb[:, 0, :]  # CLS token

        emb = emb.cpu().numpy().astype(np.float32)
        all_embeddings.append(emb)

    result = np.concatenate(all_embeddings, axis=0)

    # L2-normalize
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    result = result / norms
    return result


def _conch_zero_shot_score(
    tile_embeddings: np.ndarray,
    text_embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute per-tile staining score via cosine similarity with text prompts.

    Combines similarities with the staining-class prompts into a single 0-1
    score per tile.  Higher = stronger staining presence.

    Args:
        tile_embeddings: L2-normalized (n_tiles, 512).
        text_embeddings: Dict mapping class name to (1, 512) embeddings.

    Returns:
        numpy array of shape (n_tiles,) with scores in [0, 1].
    """
    # Compute cosine similarities (dot product since both are L2-normed)
    sims: dict[str, np.ndarray] = {}
    for cls_name, text_emb in text_embeddings.items():
        sims[cls_name] = (tile_embeddings @ text_emb.T).squeeze(-1)

    # Weighted combination: higher staining classes get higher weight
    weights = {
        "absent_staining": 0.0,
        "weak_staining": 0.33,
        "moderate_staining": 0.67,
        "strong_staining": 1.0,
    }

    # Softmax over classes to get pseudo-probabilities per tile, then
    # compute weighted average as score.
    all_sims = np.stack([sims[k] for k in weights], axis=-1)  # (n_tiles, 4)
    # Temperature-scaled softmax
    temperature = 0.07
    exp_sims = np.exp(all_sims / temperature)
    probs = exp_sims / exp_sims.sum(axis=-1, keepdims=True)

    w = np.array([weights[k] for k in weights])
    scores = (probs * w[None, :]).sum(axis=-1)

    # Clip to [0, 1]
    scores = np.clip(scores, 0.0, 1.0)
    return scores


# ---------------------------------------------------------------------------
# Pure-vision fallback: approximate CONCH zero-shot with hand-crafted features
# ---------------------------------------------------------------------------

def _membrane_staining_score(
    dab_channel: np.ndarray,
    nucleus_masks: np.ndarray,
) -> float:
    """Score how much the DAB signal looks like membrane staining.

    Membrane staining has a characteristic ring pattern: DAB is concentrated
    at cell boundaries, not uniformly across the cell body.

    Adapted from filter_branch_f.py.

    Returns:
        Score in [0, 1], higher = more membrane-like DAB pattern.
    """
    if nucleus_masks.max() == 0:
        return 0.0

    ring_scores: list[float] = []
    cell_ids = np.unique(nucleus_masks)
    cell_ids = cell_ids[cell_ids > 0]

    for nid in cell_ids[:50]:  # sample max 50 cells per tile for speed
        nuc_mask = (nucleus_masks == nid).astype(np.uint8)

        # Create ring: dilate nucleus by 5px, subtract nucleus
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(nuc_mask, kernel, iterations=1)
        ring = dilated.astype(bool) & ~nuc_mask.astype(bool)

        if ring.sum() < 10 or nuc_mask.sum() < 10:
            continue

        dab_ring = dab_channel[ring].mean()
        dab_nucleus = dab_channel[nuc_mask.astype(bool)].mean()

        if dab_ring + dab_nucleus > 0.01:
            ring_ratio = dab_ring / (dab_ring + dab_nucleus)
            ring_scores.append(float(ring_ratio))

    if not ring_scores:
        return 0.0

    mean_ratio = float(np.mean(ring_scores))
    dab_presence = min(float(dab_channel.mean()) / 0.15, 1.0)

    return float(mean_ratio * dab_presence)


def _compute_cell_density(nucleus_masks: np.ndarray) -> float:
    """Compute cell density (cells per 10000 pixels).

    Epithelial tissue typically has higher cell density than stroma.

    Args:
        nucleus_masks: Integer label mask (H, W).

    Returns:
        Cells per 10000 pixels.
    """
    n_cells = len(np.unique(nucleus_masks)) - (1 if 0 in nucleus_masks else 0)
    total_px = nucleus_masks.size
    return n_cells / total_px * 10000 if total_px > 0 else 0.0


def _vision_tile_score(
    tile_rgb: np.ndarray,
    dab_channel: np.ndarray,
    nucleus_masks: np.ndarray,
) -> float:
    """Pure-vision approximation of CONCH zero-shot staining score.

    Combines four features that capture what the CONCH text prompts describe:
      1. DAB channel mean intensity  (higher = more staining)
      2. Ring-to-interior DAB ratio  (higher = membrane pattern)
      3. Cell density                (epithelial tissue has more cells)
      4. Color saturation            (DAB tissue has higher saturation)

    Args:
        tile_rgb: RGB image, uint8 (H, W, 3).
        dab_channel: DAB concentration map (H, W) float32.
        nucleus_masks: Integer nucleus label mask (H, W).

    Returns:
        Score in [0, 1], higher = stronger staining presence.
    """
    # Feature 1: DAB channel mean intensity (normalized to ~[0, 1])
    # Typical DAB range is 0-0.5; divide by 0.3 to spread the range
    dab_score = min(float(dab_channel.mean()) / 0.30, 1.0)

    # Feature 2: Membrane ring pattern score (from _membrane_staining_score)
    ring_score = _membrane_staining_score(dab_channel, nucleus_masks)

    # Feature 3: Cell density (normalized; ~5-20 cells/10kpx is typical for epithelial)
    density = _compute_cell_density(nucleus_masks)
    density_score = min(density / 15.0, 1.0)

    # Feature 4: Color saturation in HSV (DAB-stained tissue has browner/more saturated color)
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    sat_score = float(hsv[:, :, 1].mean()) / 255.0

    # Weighted combination
    # DAB intensity is the strongest signal; ring pattern and density provide
    # tissue-type discrimination; saturation is a weak but complementary cue.
    score = (
        0.35 * dab_score
        + 0.30 * ring_score
        + 0.20 * density_score
        + 0.15 * sat_score
    )
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Tile scoring: auto-selects CONCH or vision fallback
# ---------------------------------------------------------------------------

def _score_tiles_conch(
    data_dir: Path,
    items: list[dict],
    deconv: StainDeconvolver,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, str]:
    """Score tiles using CONCH zero-shot text similarity.

    Args:
        data_dir: Root data directory.
        items: Dataset item dicts.
        deconv: Stain deconvolver for DAB extraction.
        device: Torch device.
        batch_size: Embedding extraction batch size.

    Returns:
        scores: numpy array (n_tiles,) in [0, 1].
        method: ``"conch_zero_shot"`` or ``"conch_dab_proxy"``.
    """
    # Load tile images
    tile_images: list[np.ndarray] = []
    for item in items:
        img = tifffile.imread(data_dir / item["image"])
        tile_images.append(img)

    model, transform, tokenizer, loading_method = _load_conch_model(device)

    # Extract embeddings
    logger.info("  Extracting CONCH embeddings (%s)...", loading_method)
    embeddings = _extract_conch_embeddings(
        model, transform, tile_images, device=device,
        batch_size=batch_size, loading_method=loading_method,
    )
    logger.info("  Embeddings shape: %s", embeddings.shape)

    # Try zero-shot text scoring
    text_embeddings = _encode_text_prompts(model, tokenizer, ZERO_SHOT_PROMPTS, device)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    if text_embeddings is not None:
        logger.info("  Computing zero-shot similarity scores...")
        scores = _conch_zero_shot_score(embeddings, text_embeddings)
        return scores, "conch_zero_shot"

    # Text encoding not available — use CONCH embeddings + DAB as proxy
    logger.info("  Text encoding unavailable; using CONCH embeddings + DAB proxy...")
    from sklearn.cluster import KMeans as _KMeans

    kmeans = _KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Validate via DAB: compute per-tile mean DAB
    dab_means: list[float] = []
    for img_np in tile_images:
        dab = deconv.extract_dab(img_np)
        dab_means.append(float(dab.mean()))
    dab_arr = np.array(dab_means)

    mean_dab_0 = float(dab_arr[cluster_labels == 0].mean())
    mean_dab_1 = float(dab_arr[cluster_labels == 1].mean())
    positive_cluster = 0 if mean_dab_0 > mean_dab_1 else 1

    # Score = 1.0 for positive cluster, 0.0 for negative; smoothed by DAB
    scores = np.where(
        cluster_labels == positive_cluster,
        np.clip(dab_arr / 0.20, 0.0, 1.0),
        np.clip(dab_arr / 0.40, 0.0, 0.5),
    )
    return scores.astype(np.float32), "conch_dab_proxy"


def _score_tiles_vision(
    data_dir: Path,
    items: list[dict],
    deconv: StainDeconvolver,
) -> np.ndarray:
    """Score tiles using pure-vision approximation (no foundation model).

    Args:
        data_dir: Root data directory.
        items: Dataset item dicts.
        deconv: Stain deconvolver for DAB extraction.

    Returns:
        numpy array (n_tiles,) in [0, 1].
    """
    scores: list[float] = []
    for item in tqdm(items, desc="Vision scoring"):
        img = tifffile.imread(data_dir / item["image"])
        nuc = tifffile.imread(data_dir / item["nucleus_masks"]).astype(np.int32)
        dab = deconv.extract_dab(img)
        score = _vision_tile_score(img, dab, nuc)
        scores.append(score)
    return np.array(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main filter pipeline
# ---------------------------------------------------------------------------

def filter_dataset_branch_i(
    cfg: dict,
    output_path: Path | None = None,
    device: str = "auto",
    batch_size: int = 64,
    tile_score_threshold: float = 0.30,
) -> Path:
    """Apply Branch I filtering: CONCH-style zero-shot scoring + DAB class.

    Args:
        cfg: Pipeline configuration dictionary.
        output_path: Where to save filtered dataset. Defaults to
            ``data/segmentation_dataset_branch_i.pth``.
        device: Torch device (``"auto"``, ``"cuda"``, or ``"cpu"``).
        batch_size: Batch size for CONCH embedding extraction.
        tile_score_threshold: Minimum tile score to keep (default: 0.30).

    Returns:
        Path to the saved filtered dataset.
    """
    data_dir = Path(cfg["paths"]["data_dir"])
    sc = cfg["stain_deconvolution"]
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))

    input_path = data_dir / "segmentation_dataset_base.pth"
    if output_path is None:
        output_path = data_dir / "segmentation_dataset_branch_i.pth"

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = torch.load(input_path, weights_only=False)

    filtered: dict[str, list[dict]] = {"Train": [], "Validation": [], "Test": []}
    stats = {
        "total": 0,
        "kept": 0,
        "dropped_low_score": 0,
        "dropped_negative_dab": 0,
        "dropped_no_cells": 0,
    }

    for split in ["Train", "Validation", "Test"]:
        items = dataset[split]
        if not items:
            logger.info("Skipping empty split: %s", split)
            continue

        logger.info("Processing %s: %d items", split, len(items))

        # Step 1: Score tiles (try CONCH, fall back to vision)
        scoring_method = "vision_fallback"
        try:
            tile_scores, scoring_method = _score_tiles_conch(
                data_dir, items, deconv, device=device, batch_size=batch_size,
            )
            logger.info("  Scoring method: %s", scoring_method)
        except RuntimeError as exc:
            logger.warning("CONCH not available (%s). Using vision fallback.", exc)
            tile_scores = _score_tiles_vision(data_dir, items, deconv)
            scoring_method = "vision_fallback"
            logger.info("  Scoring method: %s", scoring_method)

        # Log score distribution
        logger.info("  Tile score stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                     tile_scores.mean(), tile_scores.std(),
                     tile_scores.min(), tile_scores.max())
        above_threshold = (tile_scores >= tile_score_threshold).sum()
        logger.info("  Tiles above threshold (%.2f): %d / %d (%.1f%%)",
                     tile_score_threshold, above_threshold, len(items),
                     above_threshold / max(len(items), 1) * 100)

        # Step 2: Filter tiles and cells
        logger.info("  Filtering tiles and cells...")
        for idx, item in enumerate(tqdm(items, desc=f"Branch I {split}")):
            stats["total"] += 1

            nuc = tifffile.imread(data_dir / item["nucleus_masks"]).astype(np.int32)
            cell = tifffile.imread(data_dir / item["cell_masks"]).astype(np.int32)

            if nuc.max() == 0:
                stats["dropped_no_cells"] += 1
                continue

            # Filter 1: Tile must score above threshold
            if tile_scores[idx] < tile_score_threshold:
                stats["dropped_low_score"] += 1
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
    logger.info("Branch I filtering complete:")
    logger.info("  Total: %d -> %d (%.1f%% kept)", stats["total"], stats["kept"],
                stats["kept"] / max(stats["total"], 1) * 100)
    logger.info("  Dropped low score: %d", stats["dropped_low_score"])
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
        description="Branch I: CONCH-style zero-shot tile filter + DAB classification.",
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
        default=64,
        help="Batch size for CONCH embedding extraction (default: 64).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Tile score threshold to keep (default: 0.30).",
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
    filter_dataset_branch_i(cfg, output_path=output_path, device=args.device,
                            batch_size=args.batch_size,
                            tile_score_threshold=args.threshold)


if __name__ == "__main__":
    main()
