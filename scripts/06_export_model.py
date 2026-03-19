"""Export a trained InstanSeg model to TorchScript for deployment.

Converts the trained model to TorchScript format, verifies the export by
loading the file back and running a forward pass on a random input, and
optionally checks attributes required for QuPath compatibility.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from instanseg_brightfield.config import get_config_hash, load_config
from instanseg_brightfield.pipeline_state import get_git_sha

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_file_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        size_bytes: File size in bytes.

    Returns:
        Human-readable size string (e.g. ``'12.3 MB'``).
    """
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"


def _find_exported_pt(folder: Path) -> Path | None:
    """Locate the exported ``.pt`` file inside an experiment folder.

    InstanSeg's ``export_to_torchscript`` writes a ``.pt`` file whose name
    matches the experiment folder.  This helper scans for it.

    Args:
        folder: Experiment folder.

    Returns:
        Path to the ``.pt`` file, or ``None`` if not found.
    """
    candidates = list(folder.glob("*.pt"))
    if len(candidates) == 1:
        return candidates[0]
    # Prefer a file whose stem matches the folder name
    for c in candidates:
        if c.stem == folder.name:
            return c
    return candidates[0] if candidates else None


def _verify_export(pt_path: Path, tile_size: int, device: str) -> bool:
    """Load the TorchScript model and run a sanity-check forward pass.

    Creates a random ``(1, 3, tile_size, tile_size)`` tensor, feeds it
    through the exported model, and verifies that the output has a valid
    spatial shape.

    Args:
        pt_path: Path to the ``.pt`` TorchScript file.
        tile_size: Spatial dimension for the test input.
        device: Device string (``'cpu'`` or ``'cuda'``).

    Returns:
        ``True`` if the sanity check passes, ``False`` otherwise.
    """
    logger.info("Loading exported model for verification: %s", pt_path)
    try:
        scripted = torch.jit.load(str(pt_path), map_location=device)
    except Exception as exc:
        logger.error("Failed to load TorchScript model: %s", exc)
        return False

    logger.info("Running sanity check with random %dx%dx3 input ...", tile_size, tile_size)
    try:
        dummy = torch.randn(1, 3, tile_size, tile_size, device=device)
        with torch.no_grad():
            output = scripted(dummy)

        # Output may be a tuple (nuclei, cells) or a single tensor
        if isinstance(output, (tuple, list)):
            shapes = [tuple(o.shape) for o in output]
            logger.info("Output shapes: %s", shapes)
            for o in output:
                if o.shape[-2] != tile_size or o.shape[-1] != tile_size:
                    logger.error(
                        "Spatial dimensions mismatch: expected %dx%d, got %s",
                        tile_size, tile_size, tuple(o.shape),
                    )
                    return False
        else:
            logger.info("Output shape: %s", tuple(output.shape))
            if output.shape[-2] != tile_size or output.shape[-1] != tile_size:
                logger.error(
                    "Spatial dimensions mismatch: expected %dx%d, got %s",
                    tile_size, tile_size, tuple(output.shape),
                )
                return False

    except Exception as exc:
        logger.error("Forward pass failed: %s", exc)
        return False

    logger.info("Sanity check passed.")
    return True


def _check_qupath_compatibility(pt_path: Path) -> bool:
    """Check whether the exported model has attributes required by QuPath.

    QuPath's InstanSeg extension expects the TorchScript model to carry
    ``pixel_size`` and ``cells_and_nuclei`` attributes.  This check is
    advisory -- a missing attribute does not prevent export but may prevent
    seamless QuPath integration.

    Args:
        pt_path: Path to the ``.pt`` TorchScript file.

    Returns:
        ``True`` if all expected attributes are present.
    """
    logger.info("Checking QuPath compatibility attributes ...")
    try:
        scripted = torch.jit.load(str(pt_path), map_location="cpu")
    except Exception as exc:
        logger.warning("Could not load model for QuPath check: %s", exc)
        return False

    all_ok = True

    # pixel_size
    if hasattr(scripted, "pixel_size"):
        logger.info("  pixel_size        : present (value=%s)", scripted.pixel_size)
    else:
        logger.warning("  pixel_size        : MISSING -- QuPath may not set the correct resolution.")
        all_ok = False

    # cells_and_nuclei
    if hasattr(scripted, "cells_and_nuclei"):
        logger.info("  cells_and_nuclei  : present (value=%s)", scripted.cells_and_nuclei)
    else:
        logger.warning("  cells_and_nuclei  : MISSING -- QuPath may not detect dual-output mode.")
        all_ok = False

    if all_ok:
        logger.info("QuPath compatibility check passed.")
    else:
        logger.warning(
            "Some QuPath attributes are missing. The model can still be used "
            "but may require manual configuration in QuPath."
        )

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained InstanSeg model to TorchScript.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        default=None,
        help="Path to the trained model folder. Defaults to models/<experiment_name>.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the verification forward pass (default: cpu).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        default=False,
        help="Skip the post-export verification step.",
    )
    parser.add_argument(
        "--skip-qupath-check",
        action="store_true",
        default=False,
        help="Skip the QuPath compatibility attribute check.",
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
    git_sha = get_git_sha()

    model_dir = Path(cfg["paths"]["model_dir"])
    experiment_name: str = cfg["training"]["experiment_name"]
    tile_size: int = cfg["tile_extraction"]["tile_size"]

    model_folder = Path(args.model_folder) if args.model_folder else model_dir / experiment_name

    logger.info("=" * 72)
    logger.info("InstanSeg Brightfield -- TorchScript Export")
    logger.info("=" * 72)
    logger.info("Config hash     : %s", cfg_hash)
    logger.info("Git SHA         : %s", git_sha)
    logger.info("Experiment name : %s", experiment_name)
    logger.info("Model folder    : %s", model_folder)
    logger.info("Model directory : %s", model_dir)

    # ------------------------------------------------------------------
    # Validate model folder
    # ------------------------------------------------------------------
    if not model_folder.exists():
        logger.error(
            "Model folder does not exist: %s. "
            "Run 04_train.py first to train the model.",
            model_folder,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Export to TorchScript
    # ------------------------------------------------------------------
    logger.info("Exporting model to TorchScript ...")
    try:
        from instanseg.utils.utils import export_to_torchscript
        export_to_torchscript(folder=experiment_name, path=str(model_dir))
    except ImportError:
        logger.error(
            "Could not import instanseg.utils.utils.export_to_torchscript. "
            "Install the package: pip install instanseg-torch[full]",
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("Export failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Locate and log the exported file
    # ------------------------------------------------------------------
    pt_path = _find_exported_pt(model_folder)
    if pt_path is None:
        # Also check the model_dir root in case the export writes there
        pt_path = _find_exported_pt(model_dir)

    if pt_path is None:
        logger.error(
            "Could not find an exported .pt file in %s or %s.",
            model_folder, model_dir,
        )
        sys.exit(1)

    file_size = pt_path.stat().st_size
    logger.info("Exported model path : %s", pt_path)
    logger.info("Exported model size : %s", _format_file_size(file_size))

    # ------------------------------------------------------------------
    # Verify the export
    # ------------------------------------------------------------------
    if not args.skip_verify:
        ok = _verify_export(pt_path, tile_size, args.device)
        if not ok:
            logger.error("Export verification FAILED.")
            sys.exit(1)
    else:
        logger.info("Skipping export verification (--skip-verify).")

    # ------------------------------------------------------------------
    # QuPath compatibility check
    # ------------------------------------------------------------------
    if not args.skip_qupath_check:
        _check_qupath_compatibility(pt_path)
    else:
        logger.info("Skipping QuPath compatibility check (--skip-qupath-check).")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("Export complete.")
    logger.info("  Model : %s", pt_path)
    logger.info("  Size  : %s", _format_file_size(file_size))
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
