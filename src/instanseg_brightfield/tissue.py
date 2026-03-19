"""Tissue detection for whole-slide images.

Identifies tissue regions to avoid extracting background-only tiles.

Adapted from Orion's tissue_detect.py with critical fixes from the
Claudin18 pipeline v3.5 investigation: the original HSV saturation Otsu
approach rejects ~84% of valid DAB-stained IHC tissue because DAB
chromogen has inherently low saturation.  The updated version uses a
relaxed brightness-only mode for IHC slides and keeps the HSV path
as an opt-in for H&E.
"""

from __future__ import annotations

import cv2
import numpy as np


def build_tissue_mask(
    thumbnail_rgb: np.ndarray,
    median_ksize: int = 7,
    brightness_threshold: int = 235,
    mode: str = "ihc",
) -> np.ndarray:
    """Build a binary tissue mask from a slide thumbnail.

    Args:
        thumbnail_rgb: RGB thumbnail of the slide (H, W, 3), uint8.
        median_ksize: Kernel size for median blur smoothing.
        brightness_threshold: Pixels brighter than this are background.
            Default raised to 235 (from 210) per Claudin18 v3.5 fix —
            weakly-stained DAB IHC tissue can have mean brightness 220-235.
        mode: Detection strategy:
            - ``"ihc"`` (default): Brightness-only filtering. Avoids the
              HSV saturation Otsu mask that rejects ~84% of DAB tissue.
            - ``"he"``: Original HSV saturation + brightness (suitable for
              H&E where saturation is reliable).

    Returns:
        Binary tissue mask (H, W), uint8, 255=tissue, 0=background.
    """
    val_smooth = cv2.medianBlur(
        cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2HSV)[:, :, 2],
        median_ksize,
    )

    if mode == "he":
        # H&E path: HSV saturation Otsu + brightness (original approach)
        sat_smooth = cv2.medianBlur(
            cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2HSV)[:, :, 1],
            median_ksize,
        )
        _, sat_mask = cv2.threshold(
            sat_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        bright_mask = (val_smooth < brightness_threshold).astype(np.uint8) * 255
        tissue_binary = np.maximum(sat_mask, bright_mask)
    else:
        # IHC path (default): brightness-only — DAB has low saturation,
        # so the Otsu mask catastrophically rejects valid tissue.
        tissue_binary = (val_smooth < brightness_threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue_binary = cv2.morphologyEx(tissue_binary, cv2.MORPH_CLOSE, kernel)
    tissue_binary = cv2.dilate(tissue_binary, kernel, iterations=2)

    return tissue_binary


def compute_tissue_fraction(
    tile_rgb: np.ndarray,
    brightness_threshold: int = 235,
    subsample: int = 8,
) -> float:
    """Quick tissue fraction estimate for a single tile.

    Args:
        tile_rgb: RGB tile (H, W, 3), uint8.
        brightness_threshold: Mean brightness above this = whitespace.
            Raised to 235 (from 220) for IHC compatibility.
        subsample: Check every Nth pixel for speed.

    Returns:
        Fraction of pixels that are tissue (0.0 to 1.0).
    """
    subsampled = tile_rgb[::subsample, ::subsample]
    mean_brightness = subsampled.mean(axis=2)
    tissue_pixels = (mean_brightness < brightness_threshold).sum()
    return float(tissue_pixels / mean_brightness.size)
