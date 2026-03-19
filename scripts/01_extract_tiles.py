"""Extract 512x512 tiles from NDPI whole-slide images, filtering for tissue regions.

Reads slides from the configured slide directory, builds tissue masks from
thumbnails, and saves tiles that meet tissue-fraction and brightness criteria.
Supports resumability via PipelineManifest so interrupted runs can continue
without re-extracting already-saved tiles.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import openslide
from tqdm import tqdm

from instanseg_brightfield.config import get_config_hash, load_config
from instanseg_brightfield.pipeline_state import PipelineManifest
from instanseg_brightfield.tissue import build_tissue_mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_best_level(slide: openslide.OpenSlide, target_mpp: float) -> tuple[int, float]:
    """Select the pyramid level whose MPP is closest to *target_mpp*.

    Args:
        slide: An open OpenSlide handle.
        target_mpp: Desired microns-per-pixel.

    Returns:
        (level_index, actual_mpp) for the chosen level.
    """
    base_mpp = float(slide.properties.get("openslide.mpp-x", 0.25))
    best_level = 0
    best_mpp = base_mpp
    best_diff = abs(base_mpp - target_mpp)

    for level in range(slide.level_count):
        level_ds = slide.level_downsamples[level]
        level_mpp = base_mpp * level_ds
        diff = abs(level_mpp - target_mpp)
        if diff < best_diff:
            best_diff = diff
            best_level = level
            best_mpp = level_mpp

    return best_level, best_mpp


def _generate_tile_positions(
    slide: openslide.OpenSlide,
    level: int,
    tile_size: int,
    padding: int = 0,
) -> list[tuple[int, int]]:
    """Generate a grid of (x, y) positions in level-0 coordinates.

    When *padding* > 0, tiles overlap by ``2 * padding`` pixels so that each
    tile's interior ``tile_size - 2 * padding`` content zone is contiguous
    with its neighbours.  This matches Orion's overlap-and-dedup tiling
    strategy.

    Args:
        slide: An open OpenSlide handle.
        level: Pyramid level to tile.
        tile_size: Tile width and height in pixels at *level*.
        padding: Overlap pixels on each side.  A value of 80 with
            tile_size=512 gives a stride of 352 and 160 px overlap.

    Returns:
        List of (x_l0, y_l0) top-left coordinates.
    """
    level_ds = slide.level_downsamples[level]
    level_w, level_h = slide.level_dimensions[level]

    # content_pixels is the non-overlap interior of each tile
    content_pixels = tile_size - 2 * padding if padding > 0 else tile_size
    step = content_pixels  # stride in level-space pixels

    positions: list[tuple[int, int]] = []
    for y in range(0, level_h - tile_size + 1, step):
        for x in range(0, level_w - tile_size + 1, step):
            x_l0 = int(x * level_ds)
            y_l0 = int(y * level_ds)
            positions.append((x_l0, y_l0))

    return positions


def _tissue_overlap(
    x_l0: int,
    y_l0: int,
    tile_size: int,
    level_ds: float,
    tissue_mask: np.ndarray,
    thumb_ds: float,
) -> float:
    """Estimate the fraction of a tile that overlaps with tissue.

    Maps the tile footprint onto the thumbnail-resolution tissue mask and
    computes the fraction of non-zero pixels.

    Args:
        x_l0: Tile x in level-0 coordinates.
        y_l0: Tile y in level-0 coordinates.
        tile_size: Tile extent in pixels at the extraction level.
        level_ds: Downsample factor for the extraction level.
        tissue_mask: Binary tissue mask at thumbnail resolution.
        thumb_ds: Downsample factor from level-0 to thumbnail.

    Returns:
        Tissue fraction in [0.0, 1.0].
    """
    # Tile extent in level-0 pixels
    extent_l0 = tile_size * level_ds

    # Map to thumbnail coordinates
    tx0 = int(x_l0 / thumb_ds)
    ty0 = int(y_l0 / thumb_ds)
    tx1 = int((x_l0 + extent_l0) / thumb_ds)
    ty1 = int((y_l0 + extent_l0) / thumb_ds)

    # Clamp to mask bounds
    th, tw = tissue_mask.shape[:2]
    tx0, tx1 = max(0, tx0), min(tw, tx1)
    ty0, ty1 = max(0, ty0), min(th, ty1)

    if tx1 <= tx0 or ty1 <= ty0:
        return 0.0

    patch = tissue_mask[ty0:ty1, tx0:tx1]
    if patch.size == 0:
        return 0.0

    return float((patch > 0).sum() / patch.size)


# ---------------------------------------------------------------------------
# Per-slide extraction
# ---------------------------------------------------------------------------

def _process_slide(
    slide_path: Path,
    tile_dir: Path,
    manifest: PipelineManifest,
    cfg_tile: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Extract tiles from a single slide.

    Args:
        slide_path: Path to the .ndpi file.
        tile_dir: Root directory for tile output (tiles go into a sub-folder
            named after the slide).
        manifest: Shared manifest for resumability.
        cfg_tile: The ``tile_extraction`` section of the config.
        rng: Seeded RNG for reproducible sampling.

    Returns:
        Metadata dict for this slide (name, mpp, level, num_tiles).
    """
    tile_size: int = cfg_tile["tile_size"]
    target_mpp: float = cfg_tile["target_mpp"]
    min_tissue: float = cfg_tile["min_tissue_fraction"]
    brightness_thresh: int = cfg_tile["brightness_threshold"]
    tiles_per_slide: int = cfg_tile["tiles_per_slide"]

    slide_name = slide_path.stem
    out_dir = tile_dir / slide_name
    out_dir.mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(str(slide_path))
    try:
        level, actual_mpp = _find_best_level(slide, target_mpp)
        level_ds = slide.level_downsamples[level]
        w_l0, h_l0 = slide.dimensions

        logger.info(
            "Slide %s: level=%d  mpp=%.3f  dims_l0=(%d, %d)",
            slide_name, level, actual_mpp, w_l0, h_l0,
        )

        # Build tissue mask from thumbnail (1:64 scale)
        thumb_ds = 64.0
        thumb = np.array(
            slide.get_thumbnail((w_l0 // 64, h_l0 // 64)).convert("RGB")
        )
        tissue_mask = build_tissue_mask(thumb)

        # Generate candidate tile positions and filter by tissue overlap
        padding: int = cfg_tile.get("padding", 0)
        all_positions = _generate_tile_positions(slide, level, tile_size, padding=padding)
        logger.info("  Total grid positions: %d", len(all_positions))

        tissue_positions: list[tuple[int, int]] = []
        for x_l0, y_l0 in all_positions:
            frac = _tissue_overlap(
                x_l0, y_l0, tile_size, level_ds, tissue_mask, thumb_ds,
            )
            if frac >= min_tissue:
                tissue_positions.append((x_l0, y_l0))

        logger.info(
            "  After tissue filter: %d (%.1f%%)",
            len(tissue_positions),
            100.0 * len(tissue_positions) / max(len(all_positions), 1),
        )

        # Sub-sample if we have more candidates than needed
        if len(tissue_positions) > tiles_per_slide:
            tissue_positions = rng.sample(tissue_positions, tiles_per_slide)

        # Extract and save tiles
        saved_count = 0
        for x_l0, y_l0 in tqdm(
            tissue_positions,
            desc=f"  {slide_name}",
            leave=False,
        ):
            tile_id = f"{slide_name}/{x_l0}_{y_l0}"
            if manifest.is_complete(tile_id):
                saved_count += 1
                continue

            region = slide.read_region((x_l0, y_l0), level, (tile_size, tile_size))
            tile_rgb = np.array(region.convert("RGB"))

            # Quick whitespace check
            if tile_rgb.mean() >= brightness_thresh:
                continue

            out_path = out_dir / f"{x_l0}_{y_l0}.png"
            tmp_path = out_dir / f".tmp_{x_l0}_{y_l0}.png"
            from PIL import Image

            Image.fromarray(tile_rgb).save(str(tmp_path))
            tmp_path.rename(out_path)
            manifest.mark_complete(tile_id, stats={
                "mean_brightness": float(tile_rgb.mean()),
            })
            saved_count += 1

        logger.info("  Saved %d tiles for %s", saved_count, slide_name)

    finally:
        slide.close()

    return {
        "slide_name": slide_name,
        "mpp": actual_mpp,
        "level": level,
        "num_tiles": saved_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract tissue tiles from NDPI whole-slide images.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--slide-dir",
        type=str,
        default=None,
        help="Override the slide directory from config.",
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
    cfg_hash = get_config_hash(cfg)
    cfg_tile = cfg["tile_extraction"]

    slide_dir = Path(args.slide_dir) if args.slide_dir else Path(cfg["paths"]["slide_dir"])
    data_dir = Path(cfg["paths"]["data_dir"])
    tile_dir = data_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Config hash: %s", cfg_hash)
    logger.info("Slide directory: %s", slide_dir)
    logger.info("Tile output directory: %s", tile_dir)

    # ------------------------------------------------------------------
    # Discover and sample slides
    # ------------------------------------------------------------------
    all_slides = sorted(
        list(slide_dir.glob("*.ndpi")) + list(slide_dir.glob("*.svs"))
    )
    if not all_slides:
        logger.error("No .ndpi files found in %s", slide_dir)
        return

    slides_to_use: int = cfg_tile["slides_to_use"]
    rng = random.Random(cfg.get("training", {}).get("seed", 42))

    if len(all_slides) > slides_to_use:
        slides = rng.sample(all_slides, slides_to_use)
    else:
        slides = list(all_slides)

    logger.info(
        "Selected %d / %d slides for extraction", len(slides), len(all_slides),
    )

    # ------------------------------------------------------------------
    # Manifest for resumability
    # ------------------------------------------------------------------
    manifest = PipelineManifest(
        path=tile_dir / "manifest.json",
        config_hash=cfg_hash,
        step_name="01_extract_tiles",
    )
    logger.info("Manifest has %d previously completed tiles", manifest.completed_count)

    # ------------------------------------------------------------------
    # Process each slide
    # ------------------------------------------------------------------
    slide_infos: list[dict[str, Any]] = []

    for slide_path in tqdm(slides, desc="Slides"):
        logger.info("Processing %s", slide_path.name)
        info = _process_slide(slide_path, tile_dir, manifest, cfg_tile, rng)
        slide_infos.append(info)
        # Save manifest after each slide for crash resilience
        manifest.save()

    # ------------------------------------------------------------------
    # Write slide_info.json
    # ------------------------------------------------------------------
    import os
    info_path = tile_dir / "slide_info.json"
    tmp_info_path = info_path.with_suffix(".json.tmp")
    with open(tmp_info_path, "w") as f:
        json.dump(slide_infos, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_info_path.rename(info_path)

    total_tiles = sum(s["num_tiles"] for s in slide_infos)
    logger.info("Done. %d total tiles across %d slides.", total_tiles, len(slide_infos))
    logger.info("Slide info written to %s", info_path)


if __name__ == "__main__":
    main()
