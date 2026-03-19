"""Generate paired nucleus + cell instance masks using self-supervised watershed segmentation.

For each extracted tile, this script:
  1. Deconvolves DAB stain from the RGB tile.
  2. Thresholds DAB to create a membrane mask (Otsu per-slide or fixed).
  3. Runs InstanSeg brightfield_nuclei model to detect nucleus instances.
  4. Grows cell instances from nuclei via marker-controlled watershed.
  5. Filters cells by quality metrics and saves 16-bit TIFF mask pairs.

Supports dual-GPU inference with VRAM-aware batch sizing, parallel CPU
post-processing via ProcessPoolExecutor, and memory backpressure to prevent
RAM blowup.  Falls back gracefully to single-GPU / CPU-only modes.

Resumable via PipelineManifest: re-running skips already-completed tiles.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import random
import sys
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch
from tqdm import tqdm

from instanseg_brightfield.config import get_config_hash, load_config
from instanseg_brightfield.dedup import deduplicate_cells
from instanseg_brightfield.pipeline_state import PipelineManifest
from instanseg_brightfield.quality import compute_tile_stats, filter_cells
from instanseg_brightfield.stain import build_stain_matrix, extract_dab
from instanseg_brightfield.watershed import (
    clean_membrane_mask,
    segment_cells,
    threshold_dab_adaptive,
    threshold_dab_fixed,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------


def _detect_gpus() -> list[torch.device]:
    """Return a list of available CUDA devices, respecting CUDA_VISIBLE_DEVICES.

    Returns an empty list when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return []
    n_gpus = torch.cuda.device_count()
    return [torch.device(f"cuda:{i}") for i in range(n_gpus)]


def _vram_aware_batch_size(
    device: torch.device,
    tile_size: int,
    requested_batch_size: int,
    reserve_mb: float = 2000.0,
    activation_factor: float = 6.0,
) -> int:
    """Compute the largest safe batch size given free VRAM on *device*.

    Args:
        device: Target CUDA device.
        tile_size: Spatial dimension of square tiles (pixels).
        requested_batch_size: Upper-bound batch size from config.
        reserve_mb: VRAM (MiB) to keep free for CUDA context / overhead.
        activation_factor: Multiplier on raw tensor size to approximate peak
            memory including activations and intermediaries (~6x empirically).

    Returns:
        An integer batch size clamped to ``[2, requested_batch_size]``.
    """
    if device.type != "cuda":
        return requested_batch_size

    props = torch.cuda.get_device_properties(device)
    free_vram_mb = (props.total_memory - torch.cuda.memory_allocated(device)) / 1024**2
    tile_mem_mb = tile_size * tile_size * 3 * 4 / 1024**2 * activation_factor
    if tile_mem_mb <= 0:
        return requested_batch_size
    safe = int((free_vram_mb - reserve_mb) / tile_mem_mb)
    return min(requested_batch_size, max(2, safe))


def _load_gpu_models(
    model_name: str,
    devices: list[torch.device],
) -> list[tuple[Any, torch.device]]:
    """Load one InstanSeg model replica per GPU device."""
    from instanseg import InstanSeg

    gpu_models: list[tuple[Any, torch.device]] = []
    for dev in devices:
        logger.info("Loading InstanSeg model '%s' on %s ...", model_name, dev)
        model = InstanSeg(model_name, device=dev)
        gpu_models.append((model, dev))
    return gpu_models


# ---------------------------------------------------------------------------
# CPU post-processing worker (runs in a child process)
# ---------------------------------------------------------------------------


def _cpu_postprocess_tile(
    tile_rgb: np.ndarray,
    nucleus_labels: np.ndarray,
    stain_matrix: np.ndarray,
    thresh_cfg: dict,
    ws_cfg: dict,
    qf_cfg: dict,
    max_cell_radius_px: float,
    otsu_threshold: float | None,
    tile_size: int,
    tile_padding: int,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Run DAB deconvolution, watershed, quality filtering for one tile.

    Designed to execute inside a :class:`ProcessPoolExecutor` worker so that
    the GIL-bound CPU work proceeds in parallel with GPU inference.

    Returns:
        ``(filtered_nuclei, filtered_cells, filter_stats, tile_stats)``
    """
    dab = extract_dab(tile_rgb, stain_matrix)

    # Threshold
    if thresh_cfg["method"] == "otsu_per_slide":
        assert otsu_threshold is not None
        membrane_raw = threshold_dab_fixed(dab, threshold=otsu_threshold)
    else:
        membrane_raw = threshold_dab_fixed(
            dab, threshold=float(thresh_cfg["fixed_threshold"])
        )

    membrane = clean_membrane_mask(
        membrane_raw,
        closing_size=int(thresh_cfg["morphology_closing_size"]),
        thin=bool(thresh_cfg["morphology_thinning"]),
    )

    cell_labels = segment_cells(
        nucleus_labels,
        dab,
        membrane,
        max_cell_radius_px=max_cell_radius_px,
        compactness=float(ws_cfg["compactness"]),
        distance_sigma=float(ws_cfg["distance_sigma"]),
    )

    filt_nuc, filt_cell, filter_stats = filter_cells(
        nucleus_labels,
        cell_labels,
        dab,
        max_cell_nucleus_ratio=float(qf_cfg["max_cell_nucleus_ratio"]),
        min_membrane_coverage=float(qf_cfg["min_membrane_coverage"]),
        min_nucleus_area_px=int(qf_cfg["min_nucleus_area_px"]),
        max_nucleus_area_px=int(qf_cfg["max_nucleus_area_px"]),
        min_cell_area_px=int(ws_cfg["min_cell_area_px"]),
    )

    # --- Overlap dedup: drop cells whose centroid is in the padding zone ---
    # Cells at tile edges will be captured as center cells by adjacent
    # overlapping tiles, so removing them here avoids truncated ground truth.
    if tile_padding > 0:
        kept_nuc_labels: set[int] = set()
        for label_id in np.unique(filt_nuc):
            if label_id == 0:
                continue
            ys, xs = np.where(filt_nuc == label_id)
            cx, cy = float(xs.mean()), float(ys.mean())
            if (tile_padding <= cx < tile_size - tile_padding
                    and tile_padding <= cy < tile_size - tile_padding):
                kept_nuc_labels.add(int(label_id))

        if kept_nuc_labels:
            # Zero out labels not in the center zone
            nuc_mask = np.isin(filt_nuc, list(kept_nuc_labels))
            filt_nuc = np.where(nuc_mask, filt_nuc, 0)
            cell_mask = np.isin(filt_cell, list(kept_nuc_labels))
            filt_cell = np.where(cell_mask, filt_cell, 0)
        else:
            filt_nuc = np.zeros_like(filt_nuc)
            filt_cell = np.zeros_like(filt_cell)

        n_removed = filter_stats.get("kept", 0) - len(kept_nuc_labels)
        filter_stats["dedup_removed"] = max(0, n_removed)
        filter_stats["dedup_kept"] = len(kept_nuc_labels)

    tile_stats = compute_tile_stats(filt_nuc, filt_cell)
    tile_stats["filter"] = filter_stats
    return filt_nuc, filt_cell, filter_stats, tile_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_tile_coords(tile_path: Path) -> tuple[int, int]:
    """Extract (x, y) coordinates from a tile filename like '1024_2048.png'."""
    stem = tile_path.stem
    parts = stem.split("_")
    return int(parts[0]), int(parts[1])


def _collect_tile_paths(tiles_dir: Path) -> dict[str, list[Path]]:
    """Group tile PNGs by slide name.

    Expected layout: ``tiles_dir/{slide_name}/{x}_{y}.png``

    Returns:
        Mapping from slide name to sorted list of tile paths.
    """
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


def _compute_otsu_threshold_for_slide(
    tile_paths: list[Path],
    stain_matrix: np.ndarray,
    max_sample_tiles: int = 50,
    seed: int = 42,
) -> float:
    """Compute Otsu threshold from pooled DAB pixels across sampled tiles.

    Args:
        tile_paths: All tile paths for one slide.
        stain_matrix: Pre-built stain matrix.
        max_sample_tiles: Number of tiles to randomly sample.
        seed: Random seed for reproducibility.

    Returns:
        Otsu threshold in DAB concentration space (float).
    """
    rng = random.Random(seed)
    sampled = (
        rng.sample(tile_paths, max_sample_tiles)
        if len(tile_paths) > max_sample_tiles
        else list(tile_paths)
    )

    all_pixels: list[np.ndarray] = []
    for tp in sampled:
        tile_rgb = cv2.cvtColor(cv2.imread(str(tp), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dab = extract_dab(tile_rgb, stain_matrix)
        # Sub-sample pixels to keep memory bounded (~10k per tile)
        flat = dab.ravel()
        if flat.size > 10_000:
            indices = rng.sample(range(flat.size), 10_000)
            flat = flat[indices]
        all_pixels.append(flat)

    pooled = np.concatenate(all_pixels)
    if pooled.size == 0:
        logger.warning("No DAB pixels found for Otsu; falling back to 0.15")
        return 0.15

    dab_max = max(float(pooled.max()), 1e-6)
    pooled_u8 = np.clip(pooled * 255.0 / dab_max, 0, 255).astype(np.uint8)
    thresh_u8, _ = cv2.threshold(pooled_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_float = float(thresh_u8) / 255.0 * dab_max
    return threshold_float


# ---------------------------------------------------------------------------
# InstanSeg batched inference
# ---------------------------------------------------------------------------


def _load_instanseg_model(
    model_name: str,
    device: torch.device,
) -> Any:
    """Load the InstanSeg model once for the entire run."""
    from instanseg import InstanSeg

    model = InstanSeg(model_name, device=device)
    return model


def _run_instanseg_batch(
    model: Any,
    tiles_rgb: list[np.ndarray],
    device: torch.device,
    use_amp: bool = True,
    batch_size: int = 8,
) -> list[np.ndarray]:
    """Run InstanSeg nucleus detection on a batch of tiles with OOM recovery.

    Args:
        model: Loaded InstanSeg model.
        tiles_rgb: List of RGB uint8 arrays (H, W, 3).
        device: Torch device.
        use_amp: Whether to use automatic mixed precision.
        batch_size: Initial batch size (halved on OOM).

    Returns:
        List of int32 nucleus label arrays, one per input tile.
    """
    results: list[np.ndarray] = [np.empty(0)] * len(tiles_rgb)
    indices = list(range(len(tiles_rgb)))
    current_bs = batch_size

    while indices:
        batches = [indices[i : i + current_bs] for i in range(0, len(indices), current_bs)]
        failed_indices: list[int] = []

        for batch_idx_list in batches:
            batch_tensors = []
            for idx in batch_idx_list:
                t = (
                    torch.from_numpy(tiles_rgb[idx])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )
                batch_tensors.append(t)

            batch_tensor = torch.cat(batch_tensors, dim=0)

            # Use pinned memory for faster host-to-device transfer
            if device.type == "cuda":
                batch_tensor = batch_tensor.pin_memory()

            try:
                with torch.inference_mode():
                    if use_amp and device.type == "cuda":
                        with torch.amp.autocast("cuda"):
                            output = model.instanseg(batch_tensor.to(device))
                    else:
                        output = model.instanseg(batch_tensor.to(device))

                output_cpu = output.cpu().numpy()
                for local_i, global_i in enumerate(batch_idx_list):
                    results[global_i] = output_cpu[local_i, 0].astype(np.int32)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_bs = max(1, current_bs // 2)
                logger.warning(
                    "CUDA OOM with batch_size=%d; retrying with batch_size=%d",
                    current_bs,
                    new_bs,
                )
                current_bs = new_bs
                failed_indices.extend(batch_idx_list)

        indices = failed_indices

    return results


# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------


def _process_slide(
    slide_name: str,
    tile_paths: list[Path],
    cfg: dict,
    stain_matrix: np.ndarray,
    gpu_models: list[tuple[Any, torch.device]],
    masks_dir: Path,
    manifest: PipelineManifest,
    slide_stats_accumulator: list[dict],
    cpu_pool: ProcessPoolExecutor,
    cpu_workers: int,
) -> None:
    """Process all tiles for a single slide.

    GPU inference is distributed round-robin across ``gpu_models`` using a
    :class:`ThreadPoolExecutor` (one thread per GPU), while CPU-heavy
    post-processing (watershed + quality filter) is offloaded to the shared
    ``cpu_pool`` (:class:`ProcessPoolExecutor`).  Memory backpressure caps
    the number of in-flight futures to avoid RAM blowup.

    Steps:
      1. Determine DAB threshold (Otsu per-slide or fixed).
      2. Batch-process tiles through InstanSeg inference (multi-GPU).
      3. Offload watershed + quality filtering to CPU workers.
      4. Save 16-bit TIFF mask pairs and update the manifest.
    """
    thresh_cfg = cfg["dab_thresholding"]
    ws_cfg = cfg["watershed"]
    qf_cfg = cfg["quality_filter"]
    nuc_cfg = cfg["nucleus_detection"]
    base_batch_size: int = nuc_cfg.get("batch_size", 8)
    use_amp: bool = nuc_cfg.get("use_amp", True)
    mpp: float = cfg["tile_extraction"].get("target_mpp", 0.5)
    tile_size: int = cfg["tile_extraction"]["tile_size"]
    tile_padding: int = cfg["tile_extraction"].get("padding", 0)
    max_cell_radius_px = ws_cfg["max_cell_radius_um"] / mpp

    n_gpus = len(gpu_models)

    # --- Threshold determination ---
    if thresh_cfg["method"] == "otsu_per_slide":
        logger.info("Computing Otsu threshold for slide %s ...", slide_name)
        otsu_threshold = _compute_otsu_threshold_for_slide(
            tile_paths, stain_matrix, max_sample_tiles=50
        )
        logger.info("Otsu threshold for %s: %.4f", slide_name, otsu_threshold)
    else:
        otsu_threshold = None  # not needed for fixed mode

    # --- Filter out already-completed tiles ---
    pending_paths: list[Path] = []
    for tp in tile_paths:
        tile_x, tile_y = _parse_tile_coords(tp)
        tile_id = f"{slide_name}/{tile_x}_{tile_y}"
        if not manifest.is_complete(tile_id):
            pending_paths.append(tp)

    if not pending_paths:
        logger.info("Slide %s: all %d tiles already complete", slide_name, len(tile_paths))
        return

    logger.info(
        "Slide %s: %d pending / %d total tiles",
        slide_name,
        len(pending_paths),
        len(tile_paths),
    )

    # --- Output directory ---
    slide_mask_dir = masks_dir / slide_name
    slide_mask_dir.mkdir(parents=True, exist_ok=True)

    # --- VRAM-aware batch sizes per GPU ---
    gpu_batch_sizes: list[int] = []
    for _model, dev in gpu_models:
        bs = _vram_aware_batch_size(dev, tile_size, base_batch_size)
        gpu_batch_sizes.append(bs)
        logger.info("GPU %s: effective batch_size=%d", dev, bs)

    # --- Memory backpressure bookkeeping ---
    MAX_INFLIGHT = max(2 * cpu_workers, 24)
    pending_futures: deque[tuple[Future, str]] = deque()

    def _drain_completed(block: bool = False) -> None:
        """Collect completed CPU futures, save masks, update manifest.

        When *block* is True, waits for the oldest future if the in-flight
        queue has reached its cap.
        """
        while pending_futures:
            if block and len(pending_futures) >= MAX_INFLIGHT:
                # Block on oldest to free memory
                fut, tid = pending_futures.popleft()
                _finalise_future(fut, tid)
            elif pending_futures[0][0].done():
                fut, tid = pending_futures.popleft()
                _finalise_future(fut, tid)
            else:
                break

    def _finalise_future(fut: Future, tile_id: str) -> None:
        """Write masks to disk and record in manifest."""
        try:
            filt_nuc, filt_cell, _fstats, tile_stats = fut.result()
        except Exception:
            logger.exception("CPU postprocess failed for %s", tile_id)
            return

        slide_stats_accumulator.append(tile_stats)

        parts = tile_id.split("/")
        coord_part = parts[-1]  # e.g. "1024_2048"
        tile_x_s, tile_y_s = coord_part.split("_")

        nuc_path = slide_mask_dir / f"{tile_x_s}_{tile_y_s}_nuclei.tiff"
        cell_path = slide_mask_dir / f"{tile_x_s}_{tile_y_s}_cells.tiff"
        nuc_tmp = slide_mask_dir / f".tmp_{tile_x_s}_{tile_y_s}_nuclei.tiff"
        cell_tmp = slide_mask_dir / f".tmp_{tile_x_s}_{tile_y_s}_cells.tiff"
        tifffile.imwrite(str(nuc_tmp), filt_nuc.astype(np.uint16), compression="zlib")
        tifffile.imwrite(str(cell_tmp), filt_cell.astype(np.uint16), compression="zlib")
        nuc_tmp.rename(nuc_path)
        cell_tmp.rename(cell_path)

        manifest.mark_complete(tile_id, stats=tile_stats)

    # --- Helper: run inference on a specific GPU ---
    def _gpu_infer(
        gpu_idx: int,
        tiles_rgb: list[np.ndarray],
    ) -> list[np.ndarray]:
        model, dev = gpu_models[gpu_idx]
        bs = gpu_batch_sizes[gpu_idx]
        return _run_instanseg_batch(
            model, tiles_rgb, device=dev, use_amp=use_amp, batch_size=bs,
        )

    # --- Process in macro-batches, distributing across GPUs ---
    # Each macro-batch is the sum of per-GPU batch sizes so every GPU
    # gets a full batch of work.
    macro_batch_size = sum(gpu_batch_sizes) if gpu_batch_sizes else base_batch_size

    with ThreadPoolExecutor(max_workers=max(n_gpus, 1)) as gpu_pool:
        for batch_start in tqdm(
            range(0, len(pending_paths), macro_batch_size),
            desc=f"  {slide_name}",
            unit="batch",
        ):
            macro_paths = pending_paths[batch_start : batch_start + macro_batch_size]

            # --- Load tile images from disk ---
            tiles_rgb: list[np.ndarray] = []
            valid_indices: list[int] = []  # indices into macro_paths that loaded OK
            for idx, tp in enumerate(macro_paths):
                img_bgr = cv2.imread(str(tp), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    logger.warning("Could not read tile %s; skipping", tp)
                    tiles_rgb.append(np.zeros((1, 1, 3), dtype=np.uint8))
                    continue
                tiles_rgb.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                valid_indices.append(idx)

            # --- Split tiles across GPUs and infer in parallel ---
            if n_gpus >= 2:
                gpu_slices: list[list[int]] = [[] for _ in range(n_gpus)]
                offset = 0
                for gi in range(n_gpus):
                    count = gpu_batch_sizes[gi]
                    gpu_slices[gi] = valid_indices[offset : offset + count]
                    offset += count
                # Any remaining tiles go to first GPU
                if offset < len(valid_indices):
                    gpu_slices[0].extend(valid_indices[offset:])

                # Submit GPU work concurrently via threads
                gpu_futures: dict[int, tuple[Future, list[int]]] = {}
                for gi, idx_list in enumerate(gpu_slices):
                    if not idx_list:
                        continue
                    sub_tiles = [tiles_rgb[i] for i in idx_list]
                    gpu_futures[gi] = (
                        gpu_pool.submit(_gpu_infer, gi, sub_tiles),
                        idx_list,
                    )

                # Gather results into the correct positions
                nucleus_labels_list: list[np.ndarray] = [
                    np.empty(0)
                ] * len(tiles_rgb)
                for gi, (fut, idx_list) in gpu_futures.items():
                    gpu_results = fut.result()
                    for local_i, global_i in enumerate(idx_list):
                        nucleus_labels_list[global_i] = gpu_results[local_i]
            else:
                # Single-GPU or CPU-only path
                model_0, dev_0 = gpu_models[0]
                bs_0 = gpu_batch_sizes[0] if gpu_batch_sizes else base_batch_size
                nucleus_labels_list = _run_instanseg_batch(
                    model_0,
                    tiles_rgb,
                    device=dev_0,
                    use_amp=use_amp and dev_0.type == "cuda",
                    batch_size=bs_0,
                )

            # --- Submit CPU post-processing for each valid tile ---
            for idx in valid_indices:
                tp = macro_paths[idx]
                tile_x, tile_y = _parse_tile_coords(tp)
                tile_id = f"{slide_name}/{tile_x}_{tile_y}"

                # Apply backpressure before submitting new work
                _drain_completed(block=True)

                fut = cpu_pool.submit(
                    _cpu_postprocess_tile,
                    tiles_rgb[idx],
                    nucleus_labels_list[idx],
                    stain_matrix,
                    thresh_cfg,
                    ws_cfg,
                    qf_cfg,
                    max_cell_radius_px,
                    otsu_threshold,
                    tile_size,
                    tile_padding,
                )
                pending_futures.append((fut, tile_id))

            # Drain any already-done futures opportunistically
            _drain_completed(block=False)

            # Persist manifest periodically for crash resilience
            manifest.save()

    # --- Drain remaining futures at end of slide ---
    while pending_futures:
        fut, tid = pending_futures.popleft()
        _finalise_future(fut, tid)
    manifest.save()


# ---------------------------------------------------------------------------
# Distribution drift monitoring
# ---------------------------------------------------------------------------


def _log_slide_distribution_warnings(
    slide_name: str,
    slide_stats: list[dict],
    global_stats: list[dict],
) -> None:
    """Log warnings if per-slide stats diverge significantly from the global running mean."""
    if len(slide_stats) < 5 or len(global_stats) < 10:
        return  # not enough data to judge

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _std(values: list[float]) -> float:
        m = _mean(values)
        return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5 if values else 0.0

    for metric in ("num_cells", "mean_nucleus_area", "mean_cell_nucleus_ratio"):
        slide_vals = [s.get(metric, 0.0) for s in slide_stats]
        global_vals = [s.get(metric, 0.0) for s in global_stats]

        slide_mean = _mean(slide_vals)
        global_mean = _mean(global_vals)
        global_sd = _std(global_vals)

        if global_sd > 0 and abs(slide_mean - global_mean) > 2.0 * global_sd:
            logger.warning(
                "Slide %s: %s mean=%.2f deviates >2 SD from global mean=%.2f (SD=%.2f)",
                slide_name,
                metric,
                slide_mean,
                global_mean,
                global_sd,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config_path: str | None = None) -> None:
    """Entry point for mask generation."""
    cfg = load_config(config_path)
    config_hash = get_config_hash(cfg)

    data_dir = Path(cfg["paths"]["data_dir"])
    tiles_dir = data_dir / "tiles"
    masks_dir = data_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Manifest for resumability
    manifest = PipelineManifest(
        masks_dir / "manifest.json",
        config_hash=config_hash,
        step_name="02_generate_masks",
    )

    # Stain matrix
    sc = cfg["stain_deconvolution"]
    stain_matrix = build_stain_matrix(
        hematoxylin=sc["hematoxylin"],
        dab=sc["dab"],
        residual=sc["residual"],
    )

    # --- GPU detection and model loading ---
    nuc_cfg = cfg["nucleus_detection"]
    device_str: str = nuc_cfg.get("device", "cuda")

    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        device_str = "cpu"

    if device_str.startswith("cuda"):
        gpu_devices = _detect_gpus()
        if not gpu_devices:
            logger.warning("No CUDA devices detected; falling back to CPU")
            gpu_devices = [torch.device("cpu")]
        n_gpus = len(gpu_devices)
        logger.info(
            "Detected %d GPU(s): %s",
            n_gpus,
            ", ".join(str(d) for d in gpu_devices),
        )
    else:
        gpu_devices = [torch.device("cpu")]
        n_gpus = 0  # signals CPU-only mode downstream

    gpu_models = _load_gpu_models(nuc_cfg["model_name"], gpu_devices)

    # --- CPU worker pool for post-processing ---
    # Use "spawn" context for CUDA safety -- fork can duplicate CUDA state.
    cpu_count = nuc_cfg.get("cpu_workers", max(1, os.cpu_count() - len(gpu_devices)))
    cpu_count = max(1, cpu_count)
    ctx = mp.get_context("spawn")
    logger.info("Starting CPU worker pool with %d workers (spawn context)", cpu_count)

    # Discover tiles grouped by slide
    slide_tiles = _collect_tile_paths(tiles_dir)
    if not slide_tiles:
        logger.error("No tiles found under %s", tiles_dir)
        sys.exit(1)

    total_tiles = sum(len(v) for v in slide_tiles.values())
    logger.info(
        "Found %d tiles across %d slides", total_tiles, len(slide_tiles)
    )

    # Global stats accumulator for distribution drift monitoring
    global_stats: list[dict] = list(manifest.get_all_stats())

    # Process one slide at a time (required for per-slide Otsu) with the
    # CPU pool kept alive across slides to amortise process creation.
    with ProcessPoolExecutor(
        max_workers=cpu_count,
        mp_context=ctx,
        max_tasks_per_child=500,
    ) as cpu_pool:
        for slide_name, tile_paths in slide_tiles.items():
            logger.info("Processing slide: %s (%d tiles)", slide_name, len(tile_paths))

            slide_stats: list[dict] = []
            _process_slide(
                slide_name=slide_name,
                tile_paths=tile_paths,
                cfg=cfg,
                stain_matrix=stain_matrix,
                gpu_models=gpu_models,
                masks_dir=masks_dir,
                manifest=manifest,
                slide_stats_accumulator=slide_stats,
                cpu_pool=cpu_pool,
                cpu_workers=cpu_count,
            )

            # Distribution drift check
            _log_slide_distribution_warnings(slide_name, slide_stats, global_stats)
            global_stats.extend(slide_stats)

    # Final save and summary
    manifest.save()
    logger.info(
        "Mask generation complete. %d tiles processed, manifest at %s",
        manifest.completed_count,
        manifest.path,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate nucleus + cell instance masks from extracted tiles.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/default.yaml).",
    )
    args = parser.parse_args()

    main(config_path=args.config)
