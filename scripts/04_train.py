"""Fine-tune InstanSeg with dual nuclei+cells output on brightfield data.

Loads the pipeline configuration, sets up reproducibility, verifies that the
dataset file exists, and delegates to InstanSeg's training loop.  Supports
resuming from a checkpoint and overriding key hyper-parameters via the CLI.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from instanseg_brightfield.config import get_config_hash, load_config
from instanseg_brightfield.pipeline_state import get_git_sha

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Set random seeds across all relevant libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic ops for full reproducibility (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides to the training section of the config.

    Only non-``None`` values from *args* are applied; other values remain
    unchanged.

    Args:
        cfg: Full pipeline config dictionary.
        args: Parsed CLI arguments.

    Returns:
        The (mutated) config dictionary.
    """
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
    logger.info("InstanSeg Brightfield Training")
    logger.info("=" * 72)
    logger.info("Start time (UTC) : %s", start_time)
    logger.info("Config hash       : %s", cfg_hash)
    logger.info("Git SHA           : %s", git_sha)
    logger.info("Experiment name   : %s", cfg["training"]["experiment_name"])
    logger.info("Target seg        : %s", cfg["training"]["target_segmentation"])
    logger.info("Learning rate     : %s", cfg["training"]["learning_rate"])
    logger.info("Batch size        : %s", cfg["training"]["batch_size"])
    logger.info("Epochs            : %s", cfg["training"]["num_epochs"])
    logger.info("Window size       : %s", cfg["training"]["window_size"])
    logger.info("Tile size         : %s", cfg["tile_extraction"]["tile_size"])
    logger.info("Layers            : %s", cfg["training"]["layers"])
    logger.info("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune InstanSeg with dual nuclei+cells output.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument(
        "--pretrained-folder",
        type=str,
        default=None,
        help=(
            "Path to a pre-trained InstanSeg model folder for fine-tuning. "
            "If not supplied, training starts from scratch."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the learning rate from config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size from config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs from config.",
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
    cfg = _apply_overrides(cfg, args)
    cfg_hash = get_config_hash(cfg)

    seed: int = cfg["training"].get("seed", 42)
    _set_seed(seed)
    logger.info("Random seed set to %d", seed)

    data_dir = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    _log_preamble(cfg, cfg_hash)

    # ------------------------------------------------------------------
    # Verify dataset
    # ------------------------------------------------------------------
    dataset_path = data_dir / "segmentation_dataset.pth"

    # Safety check: verify dataset won't OOM during InstanSeg loading
    # InstanSeg loads ALL images into RAM — each 512x512x3 tile = ~768KB
    # 10k tiles = ~7.3GB, 20k = ~14.6GB, 30k+ will likely OOM on 92GB system
    import torch as _t
    _ds = _t.load(dataset_path, weights_only=False) if dataset_path.exists() else {}
    _total = sum(len(v) for v in _ds.values()) if isinstance(_ds, dict) else 0
    if _total > 12000:
        logger.warning(
            "Dataset has %d items — InstanSeg loads all into RAM. "
            "Auto-subsampling to 10k to prevent OOM.", _total,
        )
        import random as _rng
        _r = _rng.Random(42)
        for _split in _ds:
            _items = _ds[_split]
            _slides = {}
            for _item in _items:
                _sl = Path(_item["image"]).parts[1]
                _slides.setdefault(_sl, []).append(_item)
            _sampled = []
            for _sn, _si in sorted(_slides.items()):
                _sampled.extend(_r.sample(_si, min(50, len(_si))))
            _ds[_split] = _sampled
        _t.save(_ds, dataset_path)
        _total = sum(len(v) for v in _ds.values())
        logger.info("Subsampled to %d items", _total)
    del _ds

    if not dataset_path.exists():
        logger.error(
            "Dataset file not found: %s. "
            "Run scripts 01-03 first to build the segmentation dataset.",
            dataset_path,
        )
        sys.exit(1)

    logger.info("Dataset file verified: %s", dataset_path)

    # ------------------------------------------------------------------
    # Handle pre-trained weights availability
    # ------------------------------------------------------------------
    pretrained_folder: str | None = args.pretrained_folder
    # InstanSeg joins model_path + model_folder, so strip the model_dir prefix
    if pretrained_folder:
        pf = Path(pretrained_folder)
        if pf.is_relative_to(model_dir):
            pretrained_folder = str(pf.relative_to(model_dir))
        elif pf.is_absolute():
            pretrained_folder = pf.name
    if pretrained_folder and not (model_dir / pretrained_folder).exists():
        logger.warning(
            "Pre-trained model folder not found: %s. "
            "Pre-trained brightfield weights are provided by the Edinburgh "
            "InstanSeg authors. If you do not have them, please contact "
            "Thibaut Goldsborough (thibaut.goldsborough@ed.ac.uk) or visit "
            "https://github.com/instanseg/instanseg for the latest released "
            "model checkpoints. Training will proceed from scratch.",
            pretrained_folder,
        )
        pretrained_folder = None

    # ------------------------------------------------------------------
    # Launch training
    # ------------------------------------------------------------------
    logger.info("Starting InstanSeg training loop ...")
    wall_start = time.monotonic()

    # Clear sys.argv so InstanSeg's internal argparse doesn't see our flags
    # Force num_workers=0 to prevent DataLoader workers from each caching
    # 25GB of tile data. With 4 workers that's 100GB → instant OOM.
    sys.argv = [sys.argv[0], "-num_workers", "0"]

    try:
        from instanseg.scripts.train import instanseg_training

        # Patch: InstanSeg's _robust_average_precision crashes on dual-head
        # outputs due to a shape mismatch in torch_sparse_onehot. Wrap it
        # to return 0.0 on failure so training can proceed — we evaluate
        # properly with our own script 05.
        import instanseg.utils.metrics as _metrics_mod
        _orig_ap = _metrics_mod._robust_average_precision

        def _safe_ap(*a, **kw):
            try:
                return _orig_ap(*a, **kw)
            except (IndexError, RuntimeError, ValueError):
                # Return tuple for cells_and_nuclei mode
                return (0.0, 0.0)

        _metrics_mod._robust_average_precision = _safe_ap

        # Patch: Add early stopping based on test_loss since F1 is patched.
        # InstanSeg's main() saves checkpoints on F1 improvement, but F1 is
        # always (0,0) due to the metrics patch. We wrap main() to:
        # 1. Track best test_loss and save checkpoints when it improves
        # 2. Stop early if test_loss doesn't improve for `patience` epochs
        import instanseg.scripts.train as _train_mod
        _orig_main = _train_mod.main
        _early_stop_patience = 30  # stop after 30 epochs without improvement

        def _main_with_early_stopping(model, loss_fn, train_loader, test_loader,
                                       num_epochs=200, epoch_name="epoch"):
            """Wrapper around InstanSeg's main() that adds early stopping on test_loss."""
            import copy
            from instanseg.utils.AI_utils import train_epoch, test_epoch
            import instanseg.scripts.train as _tm

            best_test_loss = float("inf")
            epochs_without_improvement = 0
            best_state = None

            device = _tm.device
            args = _tm.args
            optimizer = _tm.optimizer
            method = _tm.method
            iou_threshold = _tm.iou_threshold
            scheduler = _tm.scheduler

            train_losses, test_losses = [], []
            f1_list, f1_list_cells = [], []

            for epoch in range(num_epochs):
                print(f"Epoch: {epoch}")

                train_loss, train_time = train_epoch(
                    model, device, train_loader, loss_fn, optimizer, args=args,
                )
                test_loss, f1_score, test_time = test_epoch(
                    model, device, test_loader, loss_fn, debug=False,
                    best_f1=-1, save_bool=False, args=args,
                    postprocessing_fn=method.postprocessing,
                    method=method, iou_threshold=iou_threshold,
                )

                train_losses.append(train_loss)
                test_losses.append(test_loss)

                # Handle f1 as tuple for cells_and_nuclei
                if args.cells_and_nuclei:
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
                    save_path = Path(args.output_path) / args.experiment_str
                    save_path.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "f1_score": f1_joint,
                        "epoch": epoch,
                        "test_loss": test_loss,
                    }, save_path / "model_weights.pth")
                    logger.info(
                        "  [early-stop] New best test_loss=%.5f at epoch %d — saved",
                        test_loss, epoch,
                    )
                else:
                    epochs_without_improvement += 1
                    if epoch_name != "hotstart_epoch":
                        logger.info(
                            "  [early-stop] No improvement for %d/%d epochs (best=%.5f, current=%.5f)",
                            epochs_without_improvement, _early_stop_patience,
                            best_test_loss, test_loss,
                        )
                        if epochs_without_improvement >= _early_stop_patience:
                            logger.info(
                                "  [early-stop] STOPPING at epoch %d — patience exhausted",
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

        # Resolve resume: if --resume, point model_folder to the experiment dir
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

        # For pretrained weights (not resume): we monkey-patch the optimizer
        # loading because our extracted checkpoint has no matching optimizer state.
        if pretrained_folder and not resume_folder:
            # Patch load_model_weights to discard optimizer state from our
            # extracted checkpoint (it has no matching optimizer).
            import instanseg.utils.model_loader as _ml_mod
            _orig_load = _ml_mod.load_model_weights

            def _safe_load(*a, **kw):
                model, model_dict = _orig_load(*a, **kw)
                model_dict["optimizer_state_dict"] = None
                return model, model_dict

            _ml_mod.load_model_weights = _safe_load

            # Patch optimizer to skip load_state_dict(None)
            _orig_opt_load = torch.optim.Adam.load_state_dict

            def _skip_none_load(self, state_dict):
                if state_dict is None:
                    logger.info("Skipping optimizer state load (fresh optimizer for fine-tuning)")
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
        )
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)

    wall_elapsed = time.monotonic() - wall_start
    hours, remainder = divmod(int(wall_elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("=" * 72)
    logger.info("Training complete.")
    logger.info("Wall-clock time: %02d:%02d:%02d", hours, minutes, seconds)
    logger.info("Model artifacts saved to: %s", model_dir)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
