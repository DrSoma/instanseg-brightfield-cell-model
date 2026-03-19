"""Run external cell segmentation teacher models on tiles to generate noisy GT.

Teacher models (CellSAM, Cellpose cyto3, InstanSeg fluorescence) produce cell
instance masks independently.  A consensus step then merges predictions: cells
that are confirmed by at least two of the three teachers (IoU > 0.3) are kept,
and the average boundary of the agreeing teachers is used.

These consensus masks serve as noisy ground truth that can be combined with
our own watershed-based masks for stronger supervision.

Usage:
    python scripts/teacher_cell_models.py
    python scripts/teacher_cell_models.py --models cellpose,instanseg_fluoro
    python scripts/teacher_cell_models.py --models cellsam --max-tiles 50
    python scripts/teacher_cell_models.py --device cpu --max-tiles 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Protocol

import cv2
import numpy as np
import tifffile

from instanseg_brightfield.config import load_config

logger = logging.getLogger(__name__)

# All recognised teacher model names.
ALL_MODEL_NAMES = ("cellsam", "cellpose", "instanseg_fluoro")


# ---------------------------------------------------------------------------
# Teacher model protocol and implementations
# ---------------------------------------------------------------------------


class TeacherModel(Protocol):
    """Minimal interface that every teacher model adapter must satisfy."""

    name: str

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Return an integer-labelled cell instance mask (H, W) from an RGB tile."""
        ...


# -- CellSAM ---------------------------------------------------------------


class CellSAMTeacher:
    """CellSAM (Nature Methods 2025) teacher adapter.

    Install:  ``pip install cellsam``
    """

    name: str = "cellsam"

    def __init__(self) -> None:
        try:
            from cellsam import segment_cellular_image  # type: ignore[import-untyped]

            self._segment = segment_cellular_image
        except ImportError:
            raise ImportError(
                "CellSAM is not installed. Install with: pip install cellsam"
            )
        logger.info("CellSAM teacher loaded successfully")

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Run CellSAM on an RGB tile.

        Args:
            rgb: RGB image (H, W, 3), uint8.

        Returns:
            Integer-labelled cell instance mask (H, W), int32.
        """
        labels = self._segment(rgb)
        return np.asarray(labels, dtype=np.int32)


# -- Cellpose cyto3 --------------------------------------------------------


class CellposeTeacher:
    """Cellpose cyto3 (generalist cell model) teacher adapter.

    Install:  ``pip install cellpose``
    """

    name: str = "cellpose"

    def __init__(self, device: str = "cuda") -> None:
        try:
            from cellpose import models as cp_models  # type: ignore[import-untyped]

            gpu = device.startswith("cuda")
            self._model = cp_models.Cellpose(model_type="cyto3", gpu=gpu)
        except ImportError:
            raise ImportError(
                "Cellpose is not installed. Install with: pip install cellpose"
            )
        logger.info("Cellpose cyto3 teacher loaded (device=%s)", device)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Run Cellpose cyto3 on an RGB tile.

        Args:
            rgb: RGB image (H, W, 3), uint8.

        Returns:
            Integer-labelled cell instance mask (H, W), int32.
        """
        # channels=[0, 0] tells cellpose to use the grayscale image (no
        # separate nuclear channel), which is appropriate for brightfield.
        masks, _flows, _styles, _diams = self._model.eval(
            rgb, channels=[0, 0]
        )
        return np.asarray(masks, dtype=np.int32)


# -- InstanSeg fluorescence_nuclei_and_cells --------------------------------


class InstanSegFluoroTeacher:
    """InstanSeg ``fluorescence_nuclei_and_cells`` pretrained model.

    Known behaviour: ~4.9x over-detection on brightfield, but detects cells
    on 20/20 tiles tested so far.  We keep only the cell channel (index 1).
    """

    name: str = "instanseg_fluoro"

    def __init__(self, device: str = "cuda") -> None:
        try:
            from instanseg import InstanSeg  # type: ignore[import-untyped]

            self._model = InstanSeg(
                "fluorescence_nuclei_and_cells", device=device
            )
        except ImportError:
            raise ImportError(
                "instanseg-torch is not installed. "
                "Install with: pip install instanseg-torch"
            )
        self._device = device
        logger.info(
            "InstanSeg fluorescence_nuclei_and_cells teacher loaded (device=%s)",
            device,
        )

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Run InstanSeg fluorescence model and return the cell channel.

        Args:
            rgb: RGB image (H, W, 3), uint8.

        Returns:
            Integer-labelled cell instance mask (H, W), int32.
        """
        import torch

        result = self._model(rgb)
        # The model returns (nuclei_labels, cell_labels).
        if isinstance(result, tuple) and len(result) >= 2:
            cell_labels = result[1]
        else:
            cell_labels = result

        if isinstance(cell_labels, torch.Tensor):
            cell_labels = cell_labels.cpu().numpy()

        return np.asarray(cell_labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Teacher loading with graceful fallback
# ---------------------------------------------------------------------------

# Factory mapping: name -> (class, extra_kwargs_keys)
_TEACHER_FACTORIES: dict[str, type] = {
    "cellsam": CellSAMTeacher,
    "cellpose": CellposeTeacher,
    "instanseg_fluoro": InstanSegFluoroTeacher,
}


def load_teachers(
    model_names: list[str],
    device: str = "cuda",
) -> list[Any]:
    """Attempt to load each requested teacher model, skipping unavailable ones.

    Args:
        model_names: List of teacher model identifiers.
        device: Device string for GPU-capable teachers.

    Returns:
        List of successfully loaded teacher model instances.
    """
    loaded: list[Any] = []

    for name in model_names:
        name = name.strip().lower()
        if name not in _TEACHER_FACTORIES:
            logger.warning(
                "Unknown teacher model '%s'; recognised names: %s",
                name,
                list(_TEACHER_FACTORIES.keys()),
            )
            continue

        cls = _TEACHER_FACTORIES[name]
        try:
            if name == "cellsam":
                teacher = cls()
            else:
                teacher = cls(device=device)
            loaded.append(teacher)
        except ImportError as exc:
            logger.warning(
                "Skipping teacher '%s': %s",
                name,
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load teacher '%s': %s",
                name,
                exc,
            )

    return loaded


# ---------------------------------------------------------------------------
# Tile collection
# ---------------------------------------------------------------------------


def _collect_tiles_by_slide(tiles_dir: Path) -> dict[str, list[Path]]:
    """Group tile paths by slide name.

    Args:
        tiles_dir: Root tiles directory (data/tiles).

    Returns:
        Mapping from slide name to sorted list of tile PNG paths.
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


# ---------------------------------------------------------------------------
# IoU-based instance consensus
# ---------------------------------------------------------------------------


def _instance_iou_matrix(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pair-wise IoU between instances in two label masks.

    Args:
        mask_a: Integer instance mask (H, W).
        mask_b: Integer instance mask (H, W).

    Returns:
        Tuple of (iou_matrix, ids_a, ids_b) where iou_matrix has shape
        (len(ids_a), len(ids_b)).
    """
    ids_a = np.unique(mask_a)
    ids_a = ids_a[ids_a > 0]
    ids_b = np.unique(mask_b)
    ids_b = ids_b[ids_b > 0]

    if len(ids_a) == 0 or len(ids_b) == 0:
        return (
            np.zeros((len(ids_a), len(ids_b)), dtype=np.float64),
            ids_a,
            ids_b,
        )

    iou = np.zeros((len(ids_a), len(ids_b)), dtype=np.float64)
    for i, aid in enumerate(ids_a):
        a_mask = mask_a == aid
        for j, bid in enumerate(ids_b):
            b_mask = mask_b == bid
            inter = np.logical_and(a_mask, b_mask).sum()
            if inter == 0:
                continue
            union = np.logical_or(a_mask, b_mask).sum()
            iou[i, j] = inter / union

    return iou, ids_a, ids_b


def _match_instances_greedy(
    iou_matrix: np.ndarray,
    iou_threshold: float = 0.3,
) -> list[tuple[int, int, float]]:
    """Greedy 1-to-1 matching of instances between two masks.

    Each instance in mask A is matched to the highest-IoU instance in mask B
    (above *iou_threshold*) that has not already been matched.

    Args:
        iou_matrix: Shape (n_a, n_b).
        iou_threshold: Minimum IoU for a match.

    Returns:
        List of (index_a, index_b, iou_value) tuples for matched pairs.
    """
    n_a, n_b = iou_matrix.shape
    if n_a == 0 or n_b == 0:
        return []

    matches: list[tuple[int, int, float]] = []
    used_b: set[int] = set()

    # Sort by best available IoU descending for stable matching.
    best_iou_per_a = iou_matrix.max(axis=1)
    order = np.argsort(-best_iou_per_a)

    for ai in order:
        remaining = [j for j in range(n_b) if j not in used_b]
        if not remaining:
            break
        best_j = max(remaining, key=lambda j: iou_matrix[ai, j])
        if iou_matrix[ai, best_j] >= iou_threshold:
            matches.append((int(ai), int(best_j), float(iou_matrix[ai, best_j])))
            used_b.add(best_j)

    return matches


def _average_instance_mask(
    masks: list[np.ndarray],
    instance_ids: list[int],
) -> np.ndarray:
    """Compute the average (majority-vote) binary mask for one cell instance.

    For each pixel, compute the fraction of teachers that include it in this
    instance and threshold at 0.5 to produce a clean boundary.

    Args:
        masks: List of label masks from different teachers.
        instance_ids: Corresponding instance ID in each mask.

    Returns:
        Binary mask (H, W), bool.
    """
    acc = np.zeros(masks[0].shape, dtype=np.float64)
    for mask, iid in zip(masks, instance_ids):
        acc += (mask == iid).astype(np.float64)
    return acc >= (len(masks) / 2.0)


def compute_consensus(
    predictions: dict[str, np.ndarray],
    iou_threshold: float = 0.3,
    min_agreement: int = 2,
) -> np.ndarray:
    """Build an instance-level consensus mask from multiple teacher predictions.

    Algorithm:
    1. For each pair of teachers, find IoU-matched cell instances (IoU > threshold).
    2. Build a graph where each node is (teacher, instance_id) and edges connect
       matched instances across teachers.
    3. Connected components with >= ``min_agreement`` distinct teachers become
       consensus cells, with the boundary averaged across agreeing teachers.

    Args:
        predictions: Mapping from teacher name to integer instance mask (H, W).
        iou_threshold: Minimum IoU for two teacher instances to be considered
            a match.
        min_agreement: Minimum number of distinct teachers that must agree on
            a cell for it to appear in the consensus.

    Returns:
        Integer-labelled consensus cell mask (H, W), int32.
    """
    teacher_names = list(predictions.keys())
    if not teacher_names:
        return np.zeros((1, 1), dtype=np.int32)

    # Use the shape from the first prediction.
    shape = next(iter(predictions.values())).shape

    if len(teacher_names) < min_agreement:
        logger.warning(
            "Only %d teacher(s) available but min_agreement=%d; "
            "returning empty consensus",
            len(teacher_names),
            min_agreement,
        )
        return np.zeros(shape, dtype=np.int32)

    # --- Build match graph via union-find -----------------------------------

    # Each node: (teacher_name, instance_id).
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x: tuple[str, int]) -> tuple[str, int]:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: tuple[str, int], b: tuple[str, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Initialise nodes for every instance in every teacher.
    for tname, mask in predictions.items():
        for iid in np.unique(mask):
            if iid == 0:
                continue
            node = (tname, int(iid))
            parent[node] = node

    # Pair-wise matching across all teacher combinations.
    for i in range(len(teacher_names)):
        for j in range(i + 1, len(teacher_names)):
            t_a, t_b = teacher_names[i], teacher_names[j]
            mask_a, mask_b = predictions[t_a], predictions[t_b]

            iou_mat, ids_a, ids_b = _instance_iou_matrix(mask_a, mask_b)
            matches = _match_instances_greedy(iou_mat, iou_threshold)

            for ai, bi, _iou_val in matches:
                union((t_a, int(ids_a[ai])), (t_b, int(ids_b[bi])))

    # --- Collect connected components and filter by agreement ---------------

    from collections import defaultdict

    components: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for node in parent:
        components[find(node)].append(node)

    consensus = np.zeros(shape, dtype=np.int32)
    next_id = 1

    for _root, members in components.items():
        # Count distinct teachers in this component.
        distinct_teachers = {m[0] for m in members}
        if len(distinct_teachers) < min_agreement:
            continue

        # Average the boundary from all agreeing teachers.
        agreeing_masks: list[np.ndarray] = []
        agreeing_ids: list[int] = []
        for tname, iid in members:
            agreeing_masks.append(predictions[tname])
            agreeing_ids.append(iid)

        avg_mask = _average_instance_mask(agreeing_masks, agreeing_ids)

        # Write this cell into the consensus.
        consensus[avg_mask] = next_id
        next_id += 1

    return consensus


# ---------------------------------------------------------------------------
# Per-tile prediction pipeline
# ---------------------------------------------------------------------------


def predict_tile(
    rgb: np.ndarray,
    teachers: list[Any],
) -> dict[str, np.ndarray]:
    """Run all teachers on a single tile and collect their predictions.

    Args:
        rgb: RGB image (H, W, 3), uint8.
        teachers: List of loaded teacher model instances.

    Returns:
        Mapping from teacher name to integer-labelled instance mask (H, W).
    """
    results: dict[str, np.ndarray] = {}
    for teacher in teachers:
        name: str = teacher.name
        try:
            t0 = time.monotonic()
            mask = teacher.predict(rgb)
            elapsed = time.monotonic() - t0
            n_cells = int(mask.max()) if mask.size > 0 else 0
            logger.debug(
                "  %s: %d cells in %.2fs",
                name,
                n_cells,
                elapsed,
            )
            results[name] = mask
        except Exception as exc:
            logger.warning(
                "Teacher '%s' failed on tile: %s",
                name,
                exc,
            )
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_teacher_predictions(
    cfg: dict,
    teachers: list[Any],
    max_tiles: Optional[int] = None,
    compute_consensus_flag: bool = True,
    iou_threshold: float = 0.3,
    min_agreement: int = 2,
) -> None:
    """Run teacher predictions on all tiles and optionally compute consensus.

    For each tile under ``data/tiles/{slide}/*.png``:
    1. Each teacher produces a cell instance mask.
    2. Masks are saved to ``data/teacher_predictions/{model_name}/{slide}/``.
    3. If consensus is enabled and >= ``min_agreement`` teachers are available,
       a consensus mask is saved to ``data/teacher_predictions/consensus/{slide}/``.

    Args:
        cfg: Pipeline configuration dictionary.
        teachers: List of loaded teacher model instances.
        max_tiles: If set, process at most this many tiles (across all slides).
        compute_consensus_flag: Whether to compute the consensus mask.
        iou_threshold: IoU threshold for consensus matching.
        min_agreement: Minimum teacher agreement for consensus cells.
    """
    data_dir = Path(cfg["paths"]["data_dir"])
    tiles_dir = data_dir / "tiles"
    pred_dir = data_dir / "teacher_predictions"

    slide_tiles = _collect_tiles_by_slide(tiles_dir)
    if not slide_tiles:
        logger.error("No tiles found under %s", tiles_dir)
        return

    total_tiles = sum(len(v) for v in slide_tiles.values())
    logger.info(
        "Found %d tiles across %d slides",
        total_tiles,
        len(slide_tiles),
    )

    teacher_names = [t.name for t in teachers]
    logger.info("Active teachers: %s", teacher_names)

    # Pre-create output directories for each teacher and for consensus.
    for tname in teacher_names:
        (pred_dir / tname).mkdir(parents=True, exist_ok=True)
    if compute_consensus_flag and len(teachers) >= min_agreement:
        (pred_dir / "consensus").mkdir(parents=True, exist_ok=True)

    processed = 0
    total_cells: dict[str, int] = {n: 0 for n in teacher_names}
    total_cells["consensus"] = 0
    wall_start = time.monotonic()

    for slide_name, tile_paths in slide_tiles.items():
        # Create per-slide output dirs.
        for tname in teacher_names:
            (pred_dir / tname / slide_name).mkdir(parents=True, exist_ok=True)
        if compute_consensus_flag and len(teachers) >= min_agreement:
            (pred_dir / "consensus" / slide_name).mkdir(
                parents=True, exist_ok=True
            )

        for tile_path in tile_paths:
            if max_tiles is not None and processed >= max_tiles:
                break

            # Load tile.
            bgr = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
            if bgr is None:
                logger.warning("Could not read tile %s; skipping", tile_path)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            stem = tile_path.stem

            # Run each teacher.
            predictions = predict_tile(rgb, teachers)

            # Save individual teacher predictions.
            for tname, mask in predictions.items():
                out_path = (
                    pred_dir / tname / slide_name / f"{stem}_cells.tiff"
                )
                tifffile.imwrite(
                    str(out_path),
                    mask.astype(np.uint16),
                    compression="zlib",
                )
                n_cells = int(mask.max()) if mask.size > 0 else 0
                total_cells[tname] += n_cells

            # Compute and save consensus.
            if (
                compute_consensus_flag
                and len(predictions) >= min_agreement
            ):
                consensus_mask = compute_consensus(
                    predictions,
                    iou_threshold=iou_threshold,
                    min_agreement=min_agreement,
                )
                out_path = (
                    pred_dir / "consensus" / slide_name / f"{stem}_cells.tiff"
                )
                tifffile.imwrite(
                    str(out_path),
                    consensus_mask.astype(np.uint16),
                    compression="zlib",
                )
                n_consensus = int(consensus_mask.max()) if consensus_mask.size > 0 else 0
                total_cells["consensus"] += n_consensus

            processed += 1

            if processed % 10 == 0:
                elapsed = time.monotonic() - wall_start
                logger.info(
                    "Progress: %d tiles processed (%.1fs elapsed)",
                    processed,
                    elapsed,
                )

        if max_tiles is not None and processed >= max_tiles:
            logger.info("Reached --max-tiles limit (%d); stopping", max_tiles)
            break

    wall_elapsed = time.monotonic() - wall_start

    # Summary
    logger.info("=" * 72)
    logger.info("Teacher predictions complete")
    logger.info("=" * 72)
    logger.info("Tiles processed : %d", processed)
    logger.info("Wall time       : %.1fs (%.2fs/tile)", wall_elapsed,
                wall_elapsed / max(processed, 1))
    for tname in teacher_names:
        logger.info(
            "  %-20s : %d total cells",
            tname,
            total_cells[tname],
        )
    if compute_consensus_flag and len(teachers) >= min_agreement:
        logger.info(
            "  %-20s : %d total cells",
            "consensus",
            total_cells["consensus"],
        )
    logger.info("Output directory : %s", pred_dir)
    logger.info("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse arguments, load teachers, run predictions."""
    parser = argparse.ArgumentParser(
        description=(
            "Run external teacher cell segmentation models on tiles and "
            "generate consensus cell boundary predictions."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(ALL_MODEL_NAMES),
        help=(
            "Comma-separated list of teacher models to run. "
            "Choices: cellsam, cellpose, instanseg_fluoro. "
            f"Default: {','.join(ALL_MODEL_NAMES)}"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for GPU-capable models (default: cuda).",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Process at most N tiles (for quick testing).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for consensus instance matching (default: 0.3).",
    )
    parser.add_argument(
        "--min-agreement",
        type=int,
        default=2,
        help="Minimum number of teachers that must agree for consensus (default: 2).",
    )
    parser.add_argument(
        "--no-consensus",
        action="store_true",
        default=False,
        help="Skip consensus computation; save only individual teacher predictions.",
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

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    logger.info("=" * 72)
    logger.info("Teacher Cell Models")
    logger.info("=" * 72)
    logger.info("Requested models : %s", requested_models)
    logger.info("Device           : %s", args.device)
    logger.info("Max tiles        : %s", args.max_tiles or "all")
    logger.info("IoU threshold    : %.2f", args.iou_threshold)
    logger.info("Min agreement    : %d", args.min_agreement)
    logger.info("Consensus        : %s", "disabled" if args.no_consensus else "enabled")
    logger.info("=" * 72)

    # ------------------------------------------------------------------
    # Load teacher models (skip unavailable ones gracefully)
    # ------------------------------------------------------------------
    teachers = load_teachers(requested_models, device=args.device)

    if not teachers:
        logger.error(
            "No teacher models could be loaded. Install at least one of: "
            "pip install cellsam, pip install cellpose, pip install instanseg-torch"
        )
        sys.exit(1)

    logger.info(
        "Successfully loaded %d / %d requested teachers: %s",
        len(teachers),
        len(requested_models),
        [t.name for t in teachers],
    )

    # ------------------------------------------------------------------
    # Run predictions
    # ------------------------------------------------------------------
    run_teacher_predictions(
        cfg=cfg,
        teachers=teachers,
        max_tiles=args.max_tiles,
        compute_consensus_flag=not args.no_consensus,
        iou_threshold=args.iou_threshold,
        min_agreement=args.min_agreement,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
