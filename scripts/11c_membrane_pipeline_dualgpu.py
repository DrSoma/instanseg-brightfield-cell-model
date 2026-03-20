#!/usr/bin/env python3
"""Pipelined dual-GPU membrane column addition.

Architecture:
- 2 GPU workers (one per A6000), each with its own CUDA stream
- Pipelined I/O: ThreadPool pre-reads tiles while GPU processes current batch
- Batch GPU: stack multiple tiles, run stain deconv + bar-filter in one kernel
- 2 slide-level workers: process 2 slides simultaneously on 2 GPUs
- Vectorized per-cell stats via bincount (no per-cell loops)

Expected: ~2-3x faster than single-GPU sequential version.

Usage:
    python scripts/11c_membrane_pipeline_dualgpu.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

logger = logging.getLogger("membrane_dualgpu")

# --- Constants ---
STAIN_H = [0.786, 0.593, 0.174]
STAIN_DAB = [0.215, 0.422, 0.881]
STAIN_R = [0.547, -0.799, 0.249]
BAR_N = 8
BAR_KSIZE = 25
BAR_SIGMA_LONG = 5.0
BAR_SIGMA_SHORT = 1.0
GRADE_THRESHOLDS = (0.10, 0.20, 0.35)
DAB_POS_THRESH = 0.1
MIN_RING_PX = 3
RING_ERODE_K = 5
RING_ERODE_ITER = 2

DEFAULT_SLIDE_DIRS = [
    Path("/media/fernandosoto/DATA/CLDN18 slides"),
    Path("/pathodata/Claudin18_project/slides"),
]


class GPUWorker:
    """Manages GPU resources for one device."""

    def __init__(self, device_id: int):
        self.device = torch.device(f"cuda:{device_id}")
        self.stain_inv = self._build_stain_inv()
        self.bar_weights = self._build_bar_kernels()
        # Warmup
        dummy = torch.randn(4, 1, 512, 512, device=self.device)
        dummy_exp = dummy.expand(-1, BAR_N, -1, -1)
        _ = F.conv2d(dummy_exp, self.bar_weights, padding=BAR_KSIZE // 2, groups=BAR_N)
        del dummy, dummy_exp
        torch.cuda.empty_cache()

    def _build_stain_inv(self) -> torch.Tensor:
        M = np.array([STAIN_H, STAIN_DAB, STAIN_R], dtype=np.float32)
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        M = M / np.where(norms == 0, 1, norms)
        return torch.from_numpy(np.linalg.inv(M).astype(np.float32)).to(self.device)

    def _build_bar_kernels(self) -> torch.Tensor:
        half = BAR_KSIZE // 2
        y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
        kernels = []
        for i in range(BAR_N):
            theta = i * np.pi / BAR_N
            c, s = np.cos(theta), np.sin(theta)
            xr, yr = x * c + y * s, -x * s + y * c
            g = np.exp(-0.5 * (xr ** 2 / BAR_SIGMA_LONG ** 2 + yr ** 2 / BAR_SIGMA_SHORT ** 2))
            g -= g.mean()
            g /= np.abs(g).sum() + 1e-10
            kernels.append(g)
        w = np.stack(kernels)[:, None, :, :].astype(np.float32)
        return torch.from_numpy(w).to(self.device)

    def process_tile_batch(self, tiles: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Process a batch of RGB tiles -> list of (dab, bar_resp) arrays.

        Batched GPU: stain deconv + bar-filter for all tiles at once.
        """
        if not tiles:
            return []

        # Stack tiles: (B, H, W, 3)
        batch = np.stack(tiles).astype(np.float32) / 255.0
        batch_t = torch.from_numpy(batch).to(self.device).clamp(min=1 / 255)

        # Stain deconv: (B, H, W, 3) @ (3, 3).T -> take channel 1
        od = -torch.log(batch_t)
        dab_batch = (od @ self.stain_inv.T)[:, :, :, 1].clamp(min=0)  # (B, H, W)

        # Bar-filter: need (B, 1, H, W) -> expand to (B, N, H, W) for grouped conv
        dab_4d = dab_batch.unsqueeze(1)  # (B, 1, H, W)
        dab_expanded = dab_4d.expand(-1, BAR_N, -1, -1)  # (B, N, H, W)
        bar_resp = F.conv2d(dab_expanded, self.bar_weights, padding=BAR_KSIZE // 2, groups=BAR_N)
        bar_max = bar_resp.max(dim=1).values  # (B, H, W)

        # Transfer back to CPU
        dab_np = dab_batch.cpu().numpy()
        bar_np = bar_max.cpu().numpy()

        return [(dab_np[i], bar_np[i]) for i in range(len(tiles))]


def find_slide(slide_id: str, slide_dirs: list[Path]) -> Path | None:
    for d in slide_dirs:
        if not d.exists():
            continue
        for ext in (".ndpi", ".svs", ".tiff", ".tif"):
            p = d / f"{slide_id}{ext}"
            if p.exists():
                return p
    return None


def measure_cells_vectorized(
    dab: np.ndarray, bar_resp: np.ndarray, labeled: np.ndarray, max_label: int
) -> dict[str, np.ndarray]:
    """Vectorized per-cell membrane measurement using bincount."""
    mask_bin = (labeled > 0).astype(np.uint8)
    eroded = cv2.erode(mask_bin, np.ones((RING_ERODE_K, RING_ERODE_K), np.uint8), iterations=RING_ERODE_ITER)
    ring = mask_bin & (~eroded.astype(bool)).astype(np.uint8)
    ring_labels = labeled * ring

    edge_eroded = cv2.erode(mask_bin, np.ones((3, 3), np.uint8), iterations=1)
    edge = mask_bin & (~edge_eroded.astype(bool)).astype(np.uint8)
    edge_labels = labeled * edge

    flat_ring = ring_labels.ravel()
    flat_edge = edge_labels.ravel()
    flat_dab = dab.ravel()
    flat_bar = np.clip(bar_resp.ravel(), 0, None)
    ml = max_label + 1

    ring_dab_sum = np.bincount(flat_ring, weights=flat_dab * flat_bar, minlength=ml)
    ring_bar_sum = np.bincount(flat_ring, weights=flat_bar, minlength=ml)
    ring_count = np.bincount(flat_ring, minlength=ml)
    ring_dab_simple = np.bincount(flat_ring, weights=flat_dab, minlength=ml)

    with np.errstate(divide="ignore", invalid="ignore"):
        weighted_dab = np.where(ring_bar_sum > 0, ring_dab_sum / ring_bar_sum,
                                np.where(ring_count > 0, ring_dab_simple / ring_count, np.nan))
        raw_dab = np.where(ring_count > 0, ring_dab_simple / ring_count, np.nan)

    edge_count = np.bincount(flat_edge, minlength=ml)
    edge_pos = (flat_bar > 0) & (flat_dab > DAB_POS_THRESH)
    edge_pos_count = np.bincount(flat_edge, weights=edge_pos.astype(np.float64), minlength=ml)
    with np.errstate(divide="ignore", invalid="ignore"):
        completeness = np.where(edge_count > 0, edge_pos_count / edge_count, np.nan)
        thickness = np.where(edge_count > MIN_RING_PX, ring_count / np.maximum(edge_count, 1), np.nan)

    return {
        "membrane_ring_dab": weighted_dab[1:ml].astype(np.float64),
        "raw_membrane_dab": raw_dab[1:ml].astype(np.float64),
        "membrane_completeness": completeness[1:ml].astype(np.float64),
        "membrane_thickness_px": thickness[1:ml].astype(np.float64),
    }


def build_labeled_mask(polygon_wkbs, cx, cy, cell_indices, x0, y0, ds, w, h):
    """Rasterize all cell polygons into a labeled mask."""
    from shapely import from_wkb
    mask = np.zeros((h, w), dtype=np.int32)
    imap = {}
    label = 1
    for ci in cell_indices:
        wkb = polygon_wkbs[ci]
        if wkb is None:
            continue
        try:
            poly = from_wkb(wkb)
            if poly.is_empty:
                continue
            ext = np.array(poly.exterior.coords)
            local = ((ext - np.array([x0, y0])) / ds).astype(np.int32)
            cv2.fillPoly(mask, [local], label)
            imap[label] = ci
            label += 1
        except Exception:
            continue
    return mask, imap


def process_slide_pipelined(
    parquet_path: Path,
    slide_dirs: list[Path],
    gpu: GPUWorker,
    idx: int,
    total: int,
    io_pool: ThreadPoolExecutor,
    gpu_batch_size: int = 8,
) -> str:
    """Process one slide with pipelined I/O + batched GPU."""
    slide_id = parquet_path.stem.replace("_cells", "")
    prefix = f"[{idx}/{total}] {slide_id}"
    t0 = time.time()

    table = pq.read_table(parquet_path)
    n_cells = table.num_rows
    if "membrane_ring_dab" in table.column_names:
        return f"{prefix}: skip (done)"

    slide_path = find_slide(slide_id, slide_dirs)
    if slide_path is None:
        return f"{prefix}: skip (no slide)"

    import openslide
    polygon_wkbs = table.column("polygon_wkb").to_pylist()
    cx = table.column("centroid_x").to_numpy()
    cy = table.column("centroid_y").to_numpy()

    mem_dab = np.full(n_cells, np.nan, dtype=np.float64)
    raw_dab = np.full(n_cells, np.nan, dtype=np.float64)
    mem_comp = np.full(n_cells, np.nan, dtype=np.float64)
    mem_thick = np.full(n_cells, np.nan, dtype=np.float64)

    slide = openslide.OpenSlide(str(slide_path))
    try:
        base_mpp = float(slide.properties.get("openslide.mpp-x", 0.25))
        best_level = 0
        for lv in range(slide.level_count):
            if base_mpp * slide.level_downsamples[lv] <= 0.6:
                best_level = lv
        level_ds = slide.level_downsamples[best_level]
        actual_mpp = base_mpp * level_ds
        sw, sh = slide.dimensions

        tile_step = (512 - 160) * level_ds
        tile_full = 512 * level_ds
        pad_l0 = 80 * level_ds

        # Group cells into tiles
        bins: dict[tuple[int, int], list[int]] = {}
        for i in range(n_cells):
            k = (int(cx[i] / tile_step), int(cy[i] / tile_step))
            bins.setdefault(k, []).append(i)

        tile_keys = list(bins.keys())
        n_tiles = len(tile_keys)
        measured = 0

        def _read_tile(gx, gy):
            x0 = max(0, int(gx * tile_step - pad_l0))
            y0 = max(0, int(gy * tile_step - pad_l0))
            rw = min(int(tile_full + 2 * pad_l0), sw - x0)
            rh = min(int(tile_full + 2 * pad_l0), sh - y0)
            if rw <= 0 or rh <= 0:
                return None, x0, y0, 0, 0
            wlv = max(1, int(rw / level_ds))
            hlv = max(1, int(rh / level_ds))
            try:
                img = np.array(slide.read_region((x0, y0), best_level, (wlv, hlv)).convert("RGB"))
                return img, x0, y0, wlv, hlv
            except Exception:
                return None, x0, y0, 0, 0

        # Process tiles in batches with pipelined I/O
        for batch_start in range(0, n_tiles, gpu_batch_size):
            batch_end = min(batch_start + gpu_batch_size, n_tiles)
            batch_keys = tile_keys[batch_start:batch_end]

            # Pre-read all tiles in this batch using thread pool
            futures = []
            for gx, gy in batch_keys:
                futures.append(io_pool.submit(_read_tile, gx, gy))

            # Collect reads
            tile_data = []
            for f, (gx, gy) in zip(futures, batch_keys):
                img, x0, y0, wlv, hlv = f.result()
                if img is not None:
                    tile_data.append((img, x0, y0, wlv, hlv, gx, gy))

            if not tile_data:
                continue

            # Batch GPU: process all tiles at once
            # Group by size (tiles may differ at edges)
            size_groups: dict[tuple[int, int], list] = {}
            for td in tile_data:
                img, x0, y0, wlv, hlv, gx, gy = td
                key = (hlv, wlv)
                size_groups.setdefault(key, []).append(td)

            for (h, w), group in size_groups.items():
                imgs = [td[0] for td in group]

                # Resize mismatched tiles to common size
                resized = []
                for img in imgs:
                    if img.shape[0] != h or img.shape[1] != w:
                        resized.append(cv2.resize(img, (w, h)))
                    else:
                        resized.append(img)

                # Batched GPU processing
                results = gpu.process_tile_batch(resized)

                # Per-tile cell measurement
                for (dab, bar), (img, x0, y0, wlv, hlv, gx, gy) in zip(results, group):
                    cell_idxs = bins[(gx, gy)]
                    labeled, imap = build_labeled_mask(
                        polygon_wkbs, cx, cy, cell_idxs, x0, y0, level_ds, wlv, hlv
                    )
                    if not imap:
                        continue

                    max_label = max(imap.keys())
                    meas = measure_cells_vectorized(dab, bar, labeled, max_label)

                    for label_id, ci in imap.items():
                        ai = label_id - 1
                        if ai < len(meas["membrane_ring_dab"]):
                            mem_dab[ci] = meas["membrane_ring_dab"][ai]
                            raw_dab[ci] = meas["raw_membrane_dab"][ai]
                            mem_comp[ci] = meas["membrane_completeness"][ai]
                            mem_thick[ci] = meas["membrane_thickness_px"][ai]
                            measured += 1

            if (batch_start + gpu_batch_size) % (gpu_batch_size * 25) == 0:
                logger.info("%s: tile %d/%d (%d cells)", prefix, batch_end, n_tiles, measured)

    finally:
        slide.close()

    # Derived columns
    mem_thick_um = mem_thick * actual_mpp
    grades = np.zeros(n_cells, dtype=np.int8)
    t1, t2, t3 = GRADE_THRESHOLDS
    valid = np.isfinite(mem_dab)
    grades[valid & (mem_dab >= t1) & (mem_dab < t2)] = 1
    grades[valid & (mem_dab >= t2) & (mem_dab < t3)] = 2
    grades[valid & (mem_dab >= t3)] = 3

    new_table = table
    for col, arr, dt in [
        ("membrane_ring_dab", mem_dab, pa.float64()),
        ("membrane_completeness", mem_comp, pa.float64()),
        ("membrane_thickness_px", mem_thick, pa.float64()),
        ("membrane_thickness_um", mem_thick_um, pa.float64()),
        ("raw_membrane_dab", raw_dab, pa.float64()),
        ("cldn18_composite_grade", grades, pa.int8()),
        ("thresholds_calibrated", np.zeros(n_cells, dtype=bool), pa.bool_()),
    ]:
        new_table = new_table.append_column(col, pa.array(arr, type=dt))

    pq.write_table(new_table, parquet_path)

    elapsed = time.time() - t0
    pct = measured / n_cells * 100 if n_cells > 0 else 0
    return f"{prefix}: {measured:,}/{n_cells:,} cells ({pct:.0f}%), {elapsed:.1f}s"


def gpu_worker_loop(
    gpu: GPUWorker,
    slide_queue: queue.Queue,
    slide_dirs: list[Path],
    io_pool: ThreadPoolExecutor,
    gpu_batch_size: int,
    total: int,
    counter: list,  # mutable counter [done, lock]
):
    """Worker loop: pull slides from queue, process on assigned GPU."""
    while True:
        try:
            idx, pq_path = slide_queue.get(timeout=2)
        except queue.Empty:
            break

        try:
            result = process_slide_pipelined(pq_path, slide_dirs, gpu, idx, total, io_pool, gpu_batch_size)
            logger.info(result)
        except Exception as e:
            logger.error("[%d/%d] %s: ERROR %s", idx, total, pq_path.stem, e)

        with counter[1]:
            counter[0] += 1

        slide_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Pipelined dual-GPU membrane columns")
    parser.add_argument("--cohort-dir", type=Path, default=Path("/media/fernandosoto/DATA/cohort_v1/cell_data"))
    parser.add_argument("--slide-dir", nargs="*", type=Path, default=None)
    parser.add_argument("--gpu-batch", type=int, default=8, help="Tiles per GPU batch")
    parser.add_argument("--io-threads", type=int, default=16, help="I/O reader threads per GPU")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    slide_dirs = args.slide_dir if args.slide_dir else DEFAULT_SLIDE_DIRS

    # Detect available GPUs
    n_gpus = torch.cuda.device_count()
    logger.info("GPUs available: %d", n_gpus)
    if n_gpus == 0:
        logger.error("No GPUs found")
        sys.exit(1)

    # Find slides that need processing
    parquets = sorted(args.cohort_dir.glob("*_cells.parquet"))
    need = [(i, p) for i, p in enumerate(parquets)
            if "membrane_ring_dab" not in pq.read_schema(p).names]
    skip = len(parquets) - len(need)
    logger.info("Total: %d, need processing: %d, already done: %d", len(parquets), len(need), skip)

    if not need:
        logger.info("All slides already processed!")
        return

    # Initialize GPU workers
    gpus = []
    for g in range(min(n_gpus, 2)):
        logger.info("Initializing GPU %d...", g)
        gpus.append(GPUWorker(g))

    # I/O thread pool (shared across GPUs — openslide releases GIL)
    io_pool = ThreadPoolExecutor(max_workers=args.io_threads)

    # Slide queue
    sq = queue.Queue()
    for idx, (_, pq_path) in enumerate(need):
        sq.put((idx + 1, pq_path))

    total = len(need)
    counter = [0, threading.Lock()]

    t0 = time.time()

    # Launch one thread per GPU
    threads = []
    for gpu in gpus:
        t = threading.Thread(
            target=gpu_worker_loop,
            args=(gpu, sq, slide_dirs, io_pool, args.gpu_batch, total, counter),
            daemon=True,
        )
        t.start()
        threads.append(t)

    logger.info("Processing %d slides on %d GPUs (batch=%d, io_threads=%d)...",
                total, len(gpus), args.gpu_batch, args.io_threads)

    # Wait for completion
    for t in threads:
        t.join()

    io_pool.shutdown(wait=False)

    elapsed = time.time() - t0
    logger.info("DONE — %d slides in %.0fs (%.1f s/slide avg)", total, elapsed, elapsed / max(total, 1))


if __name__ == "__main__":
    main()
