#!/usr/bin/env python3
"""Package extracted tiles and masks into InstanSeg's segmentation_dataset.pth format.

Reads tile PNGs + nucleus/cell mask TIFFs produced by earlier pipeline steps,
splits by slide to avoid data leakage, builds the dataset dict, and saves it
with torch.save().
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import fastremap
import numpy as np
import tifffile
import torch
from PIL import Image
from tqdm import tqdm

from instanseg_brightfield.config import get_config_hash, load_config
from instanseg_brightfield.pipeline_state import PipelineManifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_tile_pairs(data_dir: Path) -> list[dict[str, Path]]:
    """Find all tile/mask triples under data_dir/tiles and data_dir/masks.

    Expected layout:
        data_dir/tiles/<slide_name>/<tile_id>.png
        data_dir/masks/<slide_name>/<tile_id>_nuclei.tiff
        data_dir/masks/<slide_name>/<tile_id>_cells.tiff

    Returns:
        List of dicts with keys: tile_path, nucleus_mask_path, cell_mask_path,
        slide_name, tile_id.
    """
    tiles_dir = data_dir / "tiles"
    masks_dir = data_dir / "masks"

    if not tiles_dir.exists():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    pairs: list[dict[str, Path]] = []
    for tile_path in sorted(tiles_dir.rglob("*.png")):
        slide_name = tile_path.parent.name
        tile_id = tile_path.stem

        nucleus_mask_path = masks_dir / slide_name / f"{tile_id}_nuclei.tiff"
        cell_mask_path = masks_dir / slide_name / f"{tile_id}_cells.tiff"

        if not nucleus_mask_path.exists():
            logger.warning("Missing nucleus mask for %s/%s, skipping", slide_name, tile_id)
            continue
        if not cell_mask_path.exists():
            logger.warning("Missing cell mask for %s/%s, skipping", slide_name, tile_id)
            continue

        pairs.append({
            "tile_path": tile_path,
            "nucleus_mask_path": nucleus_mask_path,
            "cell_mask_path": cell_mask_path,
            "slide_name": slide_name,
            "tile_id": tile_id,
        })

    return pairs


def _load_slide_pixel_size(data_dir: Path, slide_name: str) -> float:
    """Read pixel_size from slide_info.json saved by script 01.

    Falls back to 0.5 mpp (the pipeline target) if the file is not found.
    """
    info_path = data_dir / "tiles" / slide_name / "slide_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        return float(info.get("pixel_size", info.get("mpp", 0.5)))

    logger.warning("slide_info.json not found for %s, using default 0.5 mpp", slide_name)
    return 0.5


def _split_slides(
    slide_names: list[str],
    train_frac: float,
    val_frac: float,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split slide names into train/val/test sets.

    Args:
        slide_names: Unique slide names to split.
        train_frac: Fraction for training.
        val_frac: Fraction for validation.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping split name to list of slide names.
    """
    rng = np.random.default_rng(seed)
    names = sorted(slide_names)
    rng.shuffle(names)

    n = len(names)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    # Remaining slides go to test
    n_test = n - n_train - n_val
    if n_test < 0:
        # With very few slides, at least 1 per split
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - n_train - n_val

    return {
        "Train": list(names[:n_train]),
        "Validation": list(names[n_train : n_train + n_val]),
        "Test": list(names[n_train + n_val :]),
    }


def _validate_masks(
    nucleus_masks: np.ndarray,
    cell_masks: np.ndarray,
    image_shape: tuple[int, int],
) -> bool:
    """Check that masks are valid for the InstanSeg dataset.

    Returns True if masks pass all checks.
    """
    # Shape check
    if nucleus_masks.shape != image_shape:
        logger.warning(
            "Nucleus mask shape %s != image shape %s", nucleus_masks.shape, image_shape
        )
        return False
    if cell_masks.shape != image_shape:
        logger.warning(
            "Cell mask shape %s != image shape %s", cell_masks.shape, image_shape
        )
        return False

    # At least one instance
    if nucleus_masks.max() == 0:
        return False
    if cell_masks.max() == 0:
        return False

    return True


def _load_mask(path: Path) -> np.ndarray:
    """Load a TIFF instance mask as int32 array."""
    img = Image.open(path)
    arr = np.array(img, dtype=np.int32)
    return arr


def _build_item(
    tile_path: Path,
    nucleus_mask_path: Path,
    cell_mask_path: Path,
    pixel_size: float,
    parent_dataset: str,
    image_modality: str,
    data_dir: Path,
) -> dict | None:
    """Validate a tile + masks and build an item dict with file paths.

    Stores relative file paths (strings) instead of arrays to avoid loading
    the entire dataset into RAM. InstanSeg's data loader reads paths lazily
    via tifffile during training. The image is saved as TIFF alongside masks.

    Returns None if the tile should be skipped (invalid masks, zero instances).
    """
    import tifffile

    # Load image for validation
    image = np.array(Image.open(tile_path).convert("RGB"), dtype=np.uint8)
    h, w = image.shape[:2]

    # Load masks
    nucleus_masks = _load_mask(nucleus_mask_path)
    cell_masks = _load_mask(cell_mask_path)

    # Renumber to contiguous IDs
    nucleus_masks, _ = fastremap.renumber(nucleus_masks)
    cell_masks, _ = fastremap.renumber(cell_masks)

    # Validate
    if not _validate_masks(nucleus_masks, cell_masks, (h, w)):
        return None

    # Save image as TIFF (required for InstanSeg string-path loading)
    image_tiff_path = tile_path.with_suffix(".tiff")
    if not image_tiff_path.exists():
        tifffile.imwrite(str(image_tiff_path), image, compression="zlib")

    # Save renumbered masks (overwrite with contiguous IDs)
    tifffile.imwrite(str(nucleus_mask_path), nucleus_masks.astype(np.uint16), compression="zlib")
    tifffile.imwrite(str(cell_mask_path), cell_masks.astype(np.uint16), compression="zlib")

    # Store relative paths — InstanSeg resolves via INSTANSEG_DATASET_PATH
    return {
        "image": str(image_tiff_path.relative_to(data_dir)),
        "nucleus_masks": str(nucleus_mask_path.relative_to(data_dir)),
        "cell_masks": str(cell_mask_path.relative_to(data_dir)),
        "parent_dataset": parent_dataset,
        "image_modality": image_modality,
        "pixel_size": pixel_size,
        "licence": "Internal",
        "stain": "IHC",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_dataset(cfg: dict, output_path: Path | None = None) -> Path:
    """Build the segmentation_dataset.pth file from tiles and masks.

    Args:
        cfg: Pipeline configuration dict.
        output_path: Override for the output .pth path. Defaults to
            <data_dir>/segmentation_dataset.pth.

    Returns:
        Path to the saved dataset file.
    """
    data_dir = Path(cfg["paths"]["data_dir"])
    dataset_cfg = cfg["dataset"]

    if output_path is None:
        output_path = data_dir / "segmentation_dataset.pth"

    parent_dataset: str = dataset_cfg["parent_dataset"]
    image_modality: str = dataset_cfg["image_modality"]
    train_frac: float = dataset_cfg["train_fraction"]
    val_frac: float = dataset_cfg["val_fraction"]
    seed: int = cfg.get("training", {}).get("seed", 42)

    config_hash = get_config_hash(cfg)

    # Step 1: Discover tile/mask pairs
    logger.info("Discovering tile/mask pairs in %s ...", data_dir)
    pairs = _discover_tile_pairs(data_dir)
    if not pairs:
        raise RuntimeError(f"No tile/mask pairs found in {data_dir}")
    logger.info("Found %d tile/mask pairs", len(pairs))

    # Step 2: Read mask manifest and filter tiles with 0 cells
    mask_manifest_path = data_dir / "masks" / "manifest.json"
    skipped_zero_cells = 0
    if mask_manifest_path.exists():
        manifest = PipelineManifest(mask_manifest_path, config_hash, step_name="mask_generation")
        zero_cell_tiles: set[str] = set()
        for entry in manifest._data.get("tiles", {}).items():
            tile_id, tile_data = entry
            tile_stats = tile_data.get("stats", {})
            if tile_stats.get("num_cells", 0) == 0 or tile_stats.get("kept_instances", 0) == 0:
                zero_cell_tiles.add(tile_id)

        original_count = len(pairs)
        pairs = [p for p in pairs if p["tile_id"] not in zero_cell_tiles]
        skipped_zero_cells = original_count - len(pairs)
        if skipped_zero_cells > 0:
            logger.info(
                "Skipped %d tiles with 0 cells (from manifest); %d remaining",
                skipped_zero_cells,
                len(pairs),
            )
    else:
        logger.info("No mask manifest found at %s; skipping manifest-based filtering", mask_manifest_path)

    # Step 3: Group by slide
    slides: dict[str, list[dict]] = {}
    for pair in pairs:
        slide_name = pair["slide_name"]
        slides.setdefault(slide_name, []).append(pair)
    logger.info("Tiles span %d slides", len(slides))

    # Step 4: Split slides into train/val/test
    split_map = _split_slides(list(slides.keys()), train_frac, val_frac, seed=seed)
    for split_name, slide_list in split_map.items():
        logger.info("  %s: %d slides", split_name, len(slide_list))

    # Step 5: Cache pixel sizes per slide
    pixel_sizes: dict[str, float] = {}
    for slide_name in slides:
        pixel_sizes[slide_name] = _load_slide_pixel_size(data_dir, slide_name)

    # Step 6: Build dataset items per split
    dataset: dict[str, list[dict]] = {"Train": [], "Validation": [], "Test": []}
    total_cells: list[int] = []
    skipped_invalid = 0

    for split_name, slide_list in split_map.items():
        split_pairs = []
        for slide_name in slide_list:
            split_pairs.extend(slides.get(slide_name, []))

        logger.info("Building %s split (%d tiles) ...", split_name, len(split_pairs))
        for pair in tqdm(split_pairs, desc=split_name, unit="tile"):
            item = _build_item(
                tile_path=pair["tile_path"],
                nucleus_mask_path=pair["nucleus_mask_path"],
                cell_mask_path=pair["cell_mask_path"],
                pixel_size=pixel_sizes[pair["slide_name"]],
                parent_dataset=parent_dataset,
                image_modality=image_modality,
                data_dir=data_dir,
            )
            if item is None:
                skipped_invalid += 1
                continue

            dataset[split_name].append(item)
            total_cells.append(1)  # count valid items

    # Step 7: Log summary statistics
    total_items = sum(len(v) for v in dataset.values())
    if total_items == 0:
        raise RuntimeError("No valid tiles after filtering. Check masks and tile data.")

    mean_cells = float(np.mean(total_cells)) if total_cells else 0.0

    logger.info("--- Dataset Summary ---")
    for split_name in ("Train", "Validation", "Test"):
        logger.info("  %s: %d items", split_name, len(dataset[split_name]))
    logger.info("  Total: %d items", total_items)
    logger.info("  Mean cells per tile: %.1f", mean_cells)
    logger.info("  Skipped (zero cells from manifest): %d", skipped_zero_cells)
    logger.info("  Skipped (invalid masks): %d", skipped_invalid)

    # Step 8: Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving dataset to %s ...", output_path)
    torch.save(dataset, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved %.1f MB", file_size_mb)

    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Package tiles and masks into InstanSeg segmentation_dataset.pth",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the .pth file (default: <data_dir>/segmentation_dataset.pth)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = load_config(args.config)
    output = prepare_dataset(cfg, output_path=args.output)
    logger.info("Done. Dataset written to %s", output)


if __name__ == "__main__":
    main()
