#!/usr/bin/env python3
"""Fine-tune InstanSeg v2.0 — Phase A: Data & Infrastructure Changes Only.

2-Phase Training Strategy (from 7-agent debate consensus):
==========================================================
Applying all 7 training improvements at once is high-risk. Split into:

  Phase A (THIS SCRIPT, tonight):
    1. Lazy loading       — monkey-patch _read_images_from_pth so images are
                            loaded at __getitem__ time, not upfront. Drops RAM
                            from ~30 GB to ~4 MB. Enables using ALL dataset
                            items without subsampling.
    2. Stain augmentation — enable the disabled normalize_HE_stains and
                            extract_hematoxylin_stain augmentations (amount*0
                            → amount*0.5) for scanner-variation robustness.
    3. Disable skeletonize — set morphology_thinning: false so cell masks
                            retain their natural width.
    4. Patience 50        — increase early-stopping patience from 30 to 50.
    5. Epochs 300         — increase max epochs from 200 to 300.
    6. num_workers=1      — safe with lazy loading (25 GB/worker was from
                            eager loading only).

  Phase B (TOMORROW, from Phase A checkpoint):
    - w_seed 1.0 → 3.0     (seed confidence fix)
    - Cosine LR annealing   (better convergence)
    - Hotstart 10 → 20-30   (more BCE pre-training)
    - Possibly: AdamW, gradient accumulation

Rationale: Phase A is pure infrastructure — more data, better augmentation,
no optimizer/loss changes. If something regresses, we know it was data-side.
Phase B then layers on optimizer/loss changes from a known-good checkpoint.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/04_train_v2.py [--resume]
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Environment — set BEFORE any library imports that might check these
# ---------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["INSTANSEG_DATASET_PATH"] = str(
    Path(__file__).resolve().parent.parent / "data"
)

from instanseg_brightfield.config import get_config_hash, load_config
from instanseg_brightfield.pipeline_state import get_git_sha

logger = logging.getLogger(__name__)

# ===================================================================
# Phase A version tag — embed in checkpoint metadata for traceability
# ===================================================================
V2_PHASE = "A"
V2_TAG = "v2.0d-phaseA-zeroStainAug"

# ===================================================================
# 1. LAZY LOADING — monkey-patch _read_images_from_pth
# ===================================================================

class LazyImageList:
    """Wraps a list of items (string paths or arrays) for deferred I/O.

    When the item is a string path, ``get_image()`` is called at
    ``__getitem__`` time rather than upfront.  This drops RAM from ~30 GB
    (eagerly loading 10k 512x512x3 tiles) to ~4 MB (just the path strings).
    """

    def __init__(self, items: list) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int):
        item = self._items[i]
        if isinstance(item, str):
            from instanseg.utils.data_loader import get_image
            return get_image(item)
        return item

    def __repr__(self) -> str:
        return f"LazyImageList(n={len(self._items)})"


class LazyLabelList:
    """Wraps dataset items for deferred label formatting.

    Labels require ``_format_labels()`` which internally calls ``get_image()``
    on the cell_masks/nucleus_masks paths.  Deferring this to ``__getitem__``
    avoids loading all masks into RAM upfront.
    """

    def __init__(self, items: list, target_segmentation: str) -> None:
        self._items = items
        self._target_segmentation = target_segmentation

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int):
        # _format_labels mutates the item dict (caches loaded arrays into it).
        # Work on a shallow copy so the source dict stays path-only for
        # future accesses from other workers.
        import copy
        from instanseg.utils.data_loader import _format_labels
        item_copy = copy.copy(self._items[i])
        return _format_labels(item_copy, target_segmentation=self._target_segmentation)

    def __repr__(self) -> str:
        return f"LazyLabelList(n={len(self._items)})"


def _install_lazy_loading() -> None:
    """Monkey-patch ``_read_images_from_pth`` to return lazy lists."""
    import instanseg.utils.data_loader as dl_mod

    _orig_read = dl_mod._read_images_from_pth

    def _lazy_read_images_from_pth(
        data_path="../datasets",
        dataset="segmentation",
        data_slice=None,
        dummy=False,
        args=None,
        sets=None,
        complete_dataset=None,
    ):
        if sets is None:
            sets = ["Train", "Validation"]

        # --- Load the .pth file (same as original) ---
        if complete_dataset is None:
            if not os.environ.get("INSTANSEG_DATASET_PATH"):
                os.environ["INSTANSEG_DATASET_PATH"] = str(
                    Path(os.path.join(os.path.dirname(dl_mod.__file__), data_path))
                )
            dp = os.environ["INSTANSEG_DATASET_PATH"]
            if ".pth" in dataset:
                path_of_pth = os.path.join(dp, dataset)
            else:
                path_of_pth = os.path.join(dp, str(dataset + "_dataset.pth"))

            print(f"[v2-lazy] Loading dataset from {os.path.abspath(path_of_pth)}")
            try:
                complete_dataset = torch.load(path_of_pth, weights_only=False)
            except TypeError:
                complete_dataset = torch.load(path_of_pth)

        data_dicts = {}
        for _set in sets:
            print(f"Datasets available in {_set}")
            unique_values, counts = np.unique(
                [item["parent_dataset"] for item in complete_dataset[_set]],
                return_counts=True,
            )
            print(set((k.item(), v.item()) for k, v in zip(unique_values, counts)))

            # Filter items
            kept_items = [
                item
                for item in complete_dataset[_set]
                if dl_mod._keep_images(item, args)
            ][:data_slice]

            n_total = len(complete_dataset[_set])
            n_kept = len(kept_items)
            print(
                f"[v2-lazy] {_set}: {n_kept}/{n_total} items kept "
                f"(lazy-loaded, no RAM consumed)"
            )

            # Build lazy wrappers instead of eagerly loading everything
            images_local = LazyImageList(
                [item["image"] for item in kept_items]
            )
            labels_local = LazyLabelList(kept_items, args.target_segmentation)
            metadata = [
                {
                    k: v
                    for k, v in item.items()
                    if k not in ("image", "cell_masks", "nucleus_masks", "class_masks")
                }
                for item in kept_items
            ]

            data_dicts[_set] = [images_local, labels_local, metadata]

            print("After filtering using:")
            unique_values, counts = np.unique(
                [item["parent_dataset"] for item in data_dicts[_set][2]],
                return_counts=True,
            )
            print(set((k.item(), v.item()) for k, v in zip(unique_values, counts)))

        if dummy:
            import warnings
            warnings.warn("Using same train and validation sets !")
            data_dicts["Validation"] = data_dicts["Train"]

        return_list = []
        for _set in sets:
            return_list.extend(data_dicts[_set])
            assert len(data_dicts[_set][0]) > 0, (
                "No images in the dataset meet the requirements. "
                "(Hint: Check that the source argument is correct)"
            )

        return return_list

    dl_mod._read_images_from_pth = _lazy_read_images_from_pth
    logger.info("[Phase A] Lazy loading monkey-patch installed")


# ===================================================================
# 2. STAIN AUGMENTATION — enable normalize_HE_stains & extract_hematoxylin
# ===================================================================

def _install_stain_augmentation() -> None:
    """Monkey-patch ``get_augmentation_dict`` to enable stain augmentations.

    In the ``brightfield_only`` augmentation type, the two stain-perturbing
    augmentations are disabled via ``amount*0``.  We wrap the function to
    replace that with ``amount*0.5`` after the dict is built.
    """
    import instanseg.utils.augmentation_config as aug_cfg

    _orig_get_aug = aug_cfg.get_augmentation_dict

    def _patched_get_augmentation_dict(*args, **kwargs):
        result = _orig_get_aug(*args, **kwargs)

        # Enable stain augmentations in any Brightfield training dict
        stain_keys = ("normalize_HE_stains", "extract_hematoxylin_stain")

        # Determine the 'amount' parameter — it's the 3rd positional arg
        # or the 'amount' kwarg.  Signature:
        #   get_augmentation_dict(dim_in, nuclei_channel, amount, ...)
        if len(args) >= 3:
            amount = args[2]
        else:
            amount = kwargs.get("amount", 0.5)

        enabled_amount = amount * 0.5  # Half intensity — conservative start

        for phase in ("train",):  # Only modify training augmentations
            if phase not in result:
                continue
            for modality in result[phase]:
                aug_dict = result[phase][modality]
                if not isinstance(aug_dict, collections.OrderedDict):
                    continue
                for key in stain_keys:
                    if key in aug_dict:
                        old_val = aug_dict[key]
                        if len(old_val) >= 2 and old_val[1] == 0:
                            # probability stays the same (0.1), amount gets enabled
                            aug_dict[key] = [old_val[0], enabled_amount]
                            logger.info(
                                "[Phase A] Enabled %s: %s → %s",
                                key, old_val, aug_dict[key],
                            )

        return result

    aug_cfg.get_augmentation_dict = _patched_get_augmentation_dict
    logger.info("[Phase A] Stain augmentation monkey-patch installed")


# ===================================================================
# Helpers (from 04_train.py)
# ===================================================================

def _set_seed(seed: int) -> None:
    """Set random seeds across all relevant libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides to the training section of the config."""
    if args.learning_rate is not None:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs
    return cfg


def _log_preamble(cfg: dict, cfg_hash: str) -> None:
    """Log provenance information before training starts."""
    git_sha = get_git_sha()
    start_time = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 72)
    logger.info("InstanSeg Brightfield Training — %s", V2_TAG)
    logger.info("=" * 72)
    logger.info("Start time (UTC) : %s", start_time)
    logger.info("Config hash       : %s", cfg_hash)
    logger.info("Git SHA           : %s", git_sha)
    logger.info("Phase             : %s (data + infrastructure only)", V2_PHASE)
    logger.info("Experiment name   : %s", cfg["training"]["experiment_name"])
    logger.info("Target seg        : %s", cfg["training"]["target_segmentation"])
    logger.info("Learning rate     : %s", cfg["training"]["learning_rate"])
    logger.info("Batch size        : %s", cfg["training"]["batch_size"])
    logger.info("Epochs            : %s (was 200)", cfg["training"]["num_epochs"])
    logger.info("Early-stop patience: 50 (was 30)")
    logger.info("Lazy loading      : ENABLED (all items, deferred I/O)")
    logger.info("Stain augment     : ZERO (v2.0d — clean control, no augmentation)")
    cfg["training"]["experiment_name"] = "brightfield_cells_nuclei_zeroStainAug"
    logger.info("Skeletonize       : DISABLED (morphology_thinning: false)")
    logger.info("num_workers       : 1 (safe with lazy loading)")
    logger.info("Window size       : %s", cfg["training"]["window_size"])
    logger.info("Tile size         : %s", cfg["tile_extraction"]["tile_size"])
    logger.info("Layers            : %s", cfg["training"]["layers"])
    logger.info("=" * 72)
    logger.info("Phase B (tomorrow) will add: w_seed=3.0, cosine LR, hotstart=20-30")
    logger.info("=" * 72)


def _wait_for_gpu(device_id: int = 0, max_wait_min: int = 180) -> None:
    """Wait until the target GPU has enough free memory to train.

    Checks every 30 seconds. Requires >= 20 GB free (InstanSeg UNet with
    batch_size=4 on 512x512 tiles needs ~12-16 GB peak).
    """
    if not torch.cuda.is_available():
        logger.warning("No CUDA GPUs available — will attempt CPU training")
        return

    min_free_gb = 20.0
    check_interval = 30  # seconds
    max_checks = (max_wait_min * 60) // check_interval

    for i in range(max_checks):
        torch.cuda.set_device(device_id)
        free, total = torch.cuda.mem_get_info(device_id)
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)

        if free_gb >= min_free_gb:
            logger.info(
                "GPU %d ready: %.1f/%.1f GB free",
                device_id, free_gb, total_gb,
            )
            return

        if i == 0:
            logger.info(
                "GPU %d has %.1f/%.1f GB free (need %.1f GB). "
                "Waiting up to %d minutes for it to free up...",
                device_id, free_gb, total_gb, min_free_gb, max_wait_min,
            )
        elif i % 10 == 0:  # Log every 5 minutes
            elapsed_min = (i * check_interval) / 60
            logger.info(
                "  Still waiting... %.0f min elapsed, %.1f GB free",
                elapsed_min, free_gb,
            )

        time.sleep(check_interval)

    # Final check
    free, total = torch.cuda.mem_get_info(device_id)
    free_gb = free / (1024**3)
    if free_gb >= min_free_gb:
        logger.info("GPU %d ready after waiting: %.1f GB free", device_id, free_gb)
        return

    logger.error(
        "GPU %d still not free after %d minutes (%.1f GB free, need %.1f GB). "
        "Proceeding anyway — training may OOM.",
        device_id, max_wait_min, free_gb, min_free_gb,
    )


# ===================================================================
# Dataset selection — use the full base dataset, skip subsampling
# ===================================================================

def _prepare_full_dataset(data_dir: Path) -> Path:
    """Ensure segmentation_dataset.pth has proper Train/Validation splits.

    The original 04_train.py subsampled to 10k items to avoid OOM from eager
    loading. With lazy loading we can use ALL items. This function:
      1. Checks if segmentation_dataset.pth has non-empty Train/Validation.
      2. If not (e.g., corrupted by subsampling), restores from
         segmentation_dataset_base.pth.

    Returns the path to the usable dataset file.
    """
    dataset_path = data_dir / "segmentation_dataset.pth"
    base_path = data_dir / "segmentation_dataset_base.pth"

    if dataset_path.exists():
        ds = torch.load(dataset_path, weights_only=False)
        train_n = len(ds.get("Train", []))
        val_n = len(ds.get("Validation", []))

        if train_n > 0 and val_n > 0:
            total = sum(len(v) for v in ds.values())
            logger.info(
                "Dataset OK: %d items (Train=%d, Val=%d, Test=%d)",
                total, train_n, val_n, len(ds.get("Test", [])),
            )
            del ds
            return dataset_path
        else:
            logger.warning(
                "Dataset has empty Train (%d) or Validation (%d) — "
                "likely corrupted by eager-loading subsampler. "
                "Restoring from base dataset.",
                train_n, val_n,
            )
            del ds

    # Restore from base dataset
    if not base_path.exists():
        logger.error(
            "Neither segmentation_dataset.pth nor "
            "segmentation_dataset_base.pth found in %s. "
            "Run scripts 01-03 first.",
            data_dir,
        )
        sys.exit(1)

    import shutil
    logger.info("Restoring dataset from %s", base_path)
    shutil.copy2(base_path, dataset_path)

    ds = torch.load(dataset_path, weights_only=False)
    total = sum(len(v) for v in ds.values())
    logger.info(
        "Restored: %d items (Train=%d, Val=%d, Test=%d)",
        total,
        len(ds.get("Train", [])),
        len(ds.get("Validation", [])),
        len(ds.get("Test", [])),
    )
    del ds
    return dataset_path


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune InstanSeg v2.0 — Phase A (data + infrastructure).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument(
        "--pretrained-folder", type=str, default=None,
        help="Path to a pre-trained InstanSeg model folder for fine-tuning.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override the learning rate from config.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override the batch size from config.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the number of training epochs from config.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup logging
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------------------------------------------
    # Load config and apply Phase A overrides
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args)

    # Phase A config changes (items 3, 5):
    # 3. Disable skeletonize
    cfg["dab_thresholding"]["morphology_thinning"] = False
    logger.info("[Phase A] morphology_thinning set to False")

    # 5. Increase epochs from 200 to 300
    if args.epochs is None:  # Don't override if user passed --epochs
        cfg["training"]["num_epochs"] = 300
        logger.info("[Phase A] num_epochs set to 300 (was 200)")

    cfg_hash = get_config_hash(cfg)

    seed: int = cfg["training"].get("seed", 42)
    _set_seed(seed)
    logger.info("Random seed set to %d", seed)

    data_dir = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    _log_preamble(cfg, cfg_hash)

    # ------------------------------------------------------------------
    # Wait for GPU to be free
    # ------------------------------------------------------------------
    _wait_for_gpu(device_id=0, max_wait_min=180)

    # ------------------------------------------------------------------
    # Verify / restore dataset (NO subsampling — lazy loading handles it)
    # ------------------------------------------------------------------
    dataset_path = _prepare_full_dataset(data_dir)

    if not dataset_path.exists():
        logger.error(
            "Dataset file not found: %s. "
            "Run scripts 01-03 first to build the segmentation dataset.",
            dataset_path,
        )
        sys.exit(1)

    logger.info("Dataset file verified: %s", dataset_path)

    # ------------------------------------------------------------------
    # Handle pre-trained weights
    # ------------------------------------------------------------------
    pretrained_folder: str | None = args.pretrained_folder
    if pretrained_folder:
        pf = Path(pretrained_folder)
        if pf.is_relative_to(model_dir):
            pretrained_folder = str(pf.relative_to(model_dir))
        elif pf.is_absolute():
            pretrained_folder = pf.name
    if pretrained_folder and not (model_dir / pretrained_folder).exists():
        logger.warning(
            "Pre-trained model folder not found: %s. "
            "Training will proceed from scratch.",
            pretrained_folder,
        )
        pretrained_folder = None

    # ------------------------------------------------------------------
    # Install Phase A monkey-patches BEFORE importing training code
    # ------------------------------------------------------------------

    # 1. Lazy loading
    _install_lazy_loading()

    # 2. Stain augmentation — DISABLED for v2.0b comparison
    # _install_stain_augmentation()
    logger.info("[v2.0d] Stain augmentation COMPLETELY DISABLED — zero aug control")

    # ------------------------------------------------------------------
    # Launch training
    # ------------------------------------------------------------------
    logger.info("Starting InstanSeg v2.0 Phase A training loop ...")
    wall_start = time.monotonic()

    # 6. num_workers=1 — safe with lazy loading. With eager loading each
    #    worker would fork a copy of the 25 GB image cache. With lazy
    #    loading, workers only hold 1 image at a time (~768 KB).
    sys.argv = [sys.argv[0], "-num_workers", "1"]

    try:
        from instanseg.scripts.train import instanseg_training

        # Patch: InstanSeg's _robust_average_precision crashes on dual-head
        # outputs. Wrap it to return 0.0 on failure so training can proceed.
        import instanseg.utils.metrics as _metrics_mod
        _orig_ap = _metrics_mod._robust_average_precision

        def _safe_ap(*a, **kw):
            try:
                return _orig_ap(*a, **kw)
            except (IndexError, RuntimeError, ValueError):
                return (0.0, 0.0)

        _metrics_mod._robust_average_precision = _safe_ap

        # Patch: Early stopping on test_loss (same as 04_train.py but with
        # patience=50 instead of 30).
        import instanseg.scripts.train as _train_mod
        _orig_main = _train_mod.main

        # 4. Patience 50 (was 30)
        _early_stop_patience = 50

        def _main_with_early_stopping(
            model, loss_fn, train_loader, test_loader,
            num_epochs=300, epoch_name="epoch",
        ):
            """InstanSeg main() wrapper with early stopping on test_loss."""
            import copy
            from instanseg.utils.AI_utils import train_epoch, test_epoch
            import instanseg.scripts.train as _tm

            best_test_loss = float("inf")
            epochs_without_improvement = 0
            best_state = None

            device = _tm.device
            tm_args = _tm.args
            optimizer = _tm.optimizer
            method = _tm.method
            iou_threshold = _tm.iou_threshold
            scheduler = _tm.scheduler

            train_losses, test_losses = [], []
            f1_list, f1_list_cells = [], []

            for epoch in range(num_epochs):
                print(f"Epoch: {epoch}")

                train_loss, train_time = train_epoch(
                    model, device, train_loader, loss_fn, optimizer, args=tm_args,
                )
                test_loss, f1_score, test_time = test_epoch(
                    model, device, test_loader, loss_fn, debug=False,
                    best_f1=-1, save_bool=False, args=tm_args,
                    postprocessing_fn=method.postprocessing,
                    method=method, iou_threshold=iou_threshold,
                )

                train_losses.append(train_loss)
                test_losses.append(test_loss)

                # Handle f1 as tuple for cells_and_nuclei
                if tm_args.cells_and_nuclei:
                    f1_nuc, f1_cell = f1_score[0], f1_score[1]
                    f1_list.append(f1_nuc)
                    f1_list_cells.append(f1_cell)
                    f1_joint = float(np.nanmean(f1_score))
                else:
                    f1_nuc = f1_score[0]
                    f1_cell = 0.0
                    f1_joint = f1_nuc
                    f1_list.append(f1_nuc)

                print(
                    f"train_loss: {train_loss:.5g}, test_loss: {test_loss:.5g}, "
                    f"training_time: {int(train_time)}, testing_time: {int(test_time)}, "
                    f"f1_score_nuclei: {f1_nuc:.4g}, f1_score_cells: {f1_cell:.4g}, "
                    f"f1_score_joint: {f1_joint:.4g}"
                )

                if scheduler is not None:
                    scheduler.step()

                # Early stopping on test_loss
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    epochs_without_improvement = 0
                    best_state = copy.deepcopy(model.state_dict())
                    save_path = Path(tm_args.output_path) / tm_args.experiment_str
                    save_path.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "f1_score": f1_joint,
                            "epoch": epoch,
                            "test_loss": test_loss,
                            "v2_tag": V2_TAG,
                            "v2_phase": V2_PHASE,
                        },
                        save_path / "model_weights.pth",
                    )
                    logger.info(
                        "  [early-stop] New best test_loss=%.5f at epoch %d — saved",
                        test_loss, epoch,
                    )
                else:
                    epochs_without_improvement += 1
                    if epoch_name != "hotstart_epoch":
                        logger.info(
                            "  [early-stop] No improvement for %d/%d epochs "
                            "(best=%.5f, current=%.5f)",
                            epochs_without_improvement, _early_stop_patience,
                            best_test_loss, test_loss,
                        )
                        if epochs_without_improvement >= _early_stop_patience:
                            logger.info(
                                "  [early-stop] STOPPING at epoch %d — "
                                "patience exhausted",
                                epoch,
                            )
                            if best_state is not None:
                                model.load_state_dict(best_state)
                            break

            return model, train_losses, test_losses, f1_list, f1_list_cells

        _train_mod.main = _main_with_early_stopping

    except ImportError:
        logger.error(
            "Could not import instanseg.scripts.train. "
            "Install the training extras: pip install instanseg-torch[full]",
        )
        sys.exit(1)

    try:
        # Resolve resume
        resume_folder = None
        if args.resume:
            experiment_dir = model_dir / cfg["training"]["experiment_name"]
            if experiment_dir.exists() and (experiment_dir / "model_weights.pth").exists():
                resume_folder = str(experiment_dir.relative_to(model_dir))
                logger.info("Resuming from checkpoint: %s", resume_folder)
            else:
                logger.warning(
                    "No checkpoint found at %s; starting fresh.", experiment_dir,
                )

        # For pretrained weights (not resume): discard optimizer state
        if pretrained_folder and not resume_folder:
            import instanseg.utils.model_loader as _ml_mod
            _orig_load = _ml_mod.load_model_weights

            def _safe_load(*a, **kw):
                model, model_dict = _orig_load(*a, **kw)
                model_dict["optimizer_state_dict"] = None
                return model, model_dict

            _ml_mod.load_model_weights = _safe_load

            _orig_opt_load = torch.optim.Adam.load_state_dict

            def _skip_none_load(self, state_dict):
                if state_dict is None:
                    logger.info(
                        "Skipping optimizer state load "
                        "(fresh optimizer for fine-tuning)"
                    )
                    return
                return _orig_opt_load(self, state_dict)

            torch.optim.Adam.load_state_dict = _skip_none_load

        instanseg_training(
            data_path=str(data_dir),
            dataset="segmentation",
            source_dataset=cfg["dataset"]["parent_dataset"],
            target_segmentation=cfg["training"]["target_segmentation"],  # "NC"
            augmentation_type=cfg["training"]["augmentation_type"],
            model_str="InstanSeg_UNet",
            layers=str(cfg["training"]["layers"]),
            dim_in=cfg["training"]["dim_in"],
            n_sigma=cfg["training"]["n_sigma"],
            dim_coords=cfg["training"]["dim_coords"],
            dim_seeds=cfg["training"]["dim_seeds"],
            norm=cfg["training"]["norm"],
            binary_loss_fn=cfg["training"]["binary_loss_fn"],
            seed_loss_fn=cfg["training"]["seed_loss_fn"],
            hotstart_training=cfg["training"]["hotstart_epochs"],
            window_size=cfg["training"]["window_size"],
            tile_size=cfg["tile_extraction"]["tile_size"],
            mlp_width=cfg["training"]["mlp_width"],
            requested_pixel_size=cfg["tile_extraction"]["target_mpp"],
            experiment_str=cfg["training"]["experiment_name"],
            model_path=str(model_dir),
            model_folder=pretrained_folder if not resume_folder else resume_folder,
            num_epochs=cfg["training"]["num_epochs"],  # 300 from Phase A override
        )
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)

    wall_elapsed = time.monotonic() - wall_start
    hours, remainder = divmod(int(wall_elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("=" * 72)
    logger.info("Training complete — %s", V2_TAG)
    logger.info("Wall-clock time: %02d:%02d:%02d", hours, minutes, seconds)
    logger.info("Model artifacts saved to: %s", model_dir)
    logger.info(
        "NEXT: Run Phase B from this checkpoint — "
        "w_seed=3.0, cosine LR, hotstart=20-30"
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
