"""Marker-controlled watershed for cell segmentation.

Uses nucleus instances as seeds and DAB membrane signal as boundaries
to segment whole cells via watershed.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed


def threshold_dab_adaptive(dab_channel: np.ndarray, tissue_mask: np.ndarray | None = None) -> np.ndarray:
    """Threshold the DAB channel using Otsu's method on the tissue region.

    Per-slide adaptive thresholding avoids threshold drift across slides with
    varying staining intensity.

    Args:
        dab_channel: DAB concentration map (H, W), float32.
        tissue_mask: Optional binary mask (H, W). If provided, Otsu is computed
            only on tissue pixels. If None, uses all pixels.

    Returns:
        Binary membrane mask (H, W), uint8, 255=membrane.
    """
    if tissue_mask is not None:
        tissue_pixels = dab_channel[tissue_mask > 0]
    else:
        tissue_pixels = dab_channel.ravel()

    if tissue_pixels.size == 0:
        return np.zeros_like(dab_channel, dtype=np.uint8)

    dab_uint8 = np.clip(tissue_pixels * 255 / max(tissue_pixels.max(), 1e-6), 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(dab_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_float = threshold / 255.0 * max(dab_channel.max(), 1e-6)

    membrane = (dab_channel > threshold_float).astype(np.uint8) * 255
    return membrane


def threshold_dab_fixed(dab_channel: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """Threshold the DAB channel with a fixed value.

    Args:
        dab_channel: DAB concentration map (H, W), float32.
        threshold: DAB concentration threshold.

    Returns:
        Binary membrane mask (H, W), uint8, 255=membrane.
    """
    return (dab_channel > threshold).astype(np.uint8) * 255


def clean_membrane_mask(
    membrane: np.ndarray,
    closing_size: int = 5,
    thin: bool = True,
) -> np.ndarray:
    """Morphological cleanup of the binary membrane mask.

    Args:
        membrane: Binary membrane mask (H, W), uint8.
        closing_size: Kernel size for morphological closing (connect gaps).
        thin: Whether to thin the membrane to ~1px width.

    Returns:
        Cleaned membrane mask (H, W), uint8.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
    cleaned = cv2.morphologyEx(membrane, cv2.MORPH_CLOSE, kernel)
    if thin:
        from skimage.morphology import skeletonize
        cleaned = (skeletonize(cleaned > 0).astype(np.uint8)) * 255
    return cleaned


def segment_cells(
    nucleus_labels: np.ndarray,
    dab_channel: np.ndarray,
    membrane_mask: np.ndarray,
    max_cell_radius_px: float = 60.0,
    compactness: float = 0.0,
    distance_sigma: float = 1.0,
) -> np.ndarray:
    """Marker-controlled watershed to grow cells from nuclei using membrane boundaries.

    Args:
        nucleus_labels: Integer-labeled nucleus instance mask (H, W).
            0=background, >0=nucleus instance ID.
        dab_channel: DAB concentration map (H, W), float32.
        membrane_mask: Binary membrane mask (H, W), uint8.
        max_cell_radius_px: Maximum distance (pixels) from nucleus to cell boundary.
        compactness: Watershed compactness parameter. 0=standard watershed.
        distance_sigma: Gaussian sigma for distance transform smoothing.

    Returns:
        Integer-labeled cell instance mask (H, W). Same IDs as nucleus_labels.
    """
    if nucleus_labels.max() == 0:
        return np.zeros_like(nucleus_labels)

    # Distance transform from membrane (inverted = valleys at membranes)
    membrane_binary = membrane_mask > 0
    distance = ndimage.distance_transform_edt(~membrane_binary)

    if distance_sigma > 0:
        distance = ndimage.gaussian_filter(distance, sigma=distance_sigma)

    # Limit growth radius: zero out distances beyond max_cell_radius
    nucleus_distance = ndimage.distance_transform_edt(nucleus_labels == 0)
    distance[nucleus_distance > max_cell_radius_px] = 0

    # Watershed: nuclei are seeds, inverted distance is the landscape
    cell_labels = watershed(
        -distance,
        markers=nucleus_labels,
        mask=distance > 0,
        compactness=compactness,
    )

    return cell_labels.astype(np.int32)


def _sobel_gradient_energy(dab_channel: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude from DAB concentration.

    Used as a fallback boundary signal when DAB staining is weak or when the
    membrane mask has gaps.  Follows the same strategy as Orion's grayscale
    gradient path.

    Args:
        dab_channel: Single-channel float32 image (H, W).

    Returns:
        Gradient magnitude normalised to [0, 1], float32.
    """
    gray = dab_channel.astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    mag_max = magnitude.max()
    if mag_max > 1e-6:
        magnitude /= mag_max
    return magnitude


def _dab_weak(dab_channel: np.ndarray, threshold: float = 1e-4) -> bool:
    """Return True when the DAB signal is too weak to be useful."""
    return float(dab_channel.max()) < threshold


def segment_cells_enhanced(
    nucleus_labels: np.ndarray,
    dab_channel: np.ndarray,
    membrane_mask: np.ndarray,
    max_cell_radius_px: float = 60.0,
    compactness: float = 0.0,
    distance_sigma: float = 1.0,
    min_cell_area_px: int = 0,
    gradient_weight: float = 0.3,
    dab_weak_threshold: float = 1e-4,
) -> np.ndarray:
    """Enhanced marker-controlled watershed inspired by Orion's approach.

    Improvements over :func:`segment_cells`:

    1. **Sobel gradient fallback** -- when DAB signal is weak the energy
       landscape is built from Sobel gradient magnitude instead of DAB
       distance, exactly as Orion does when no stain matrix is available.
    2. **Combined DAB + gradient energy** -- when the membrane mask has gaps
       (sparse DAB coverage) the inverted-DAB distance is blended with the
       Sobel gradient so that edges still act as barriers even where DAB is
       absent.
    3. **Minimum cell area filtering** -- tiny cell fragments produced by
       noisy watershed ridges are removed in-place, avoiding a separate
       quality-filter pass for the most obvious artefacts.

    The function is fully backward-compatible: it accepts the same first six
    positional arguments as :func:`segment_cells` plus three optional
    keyword arguments that control the new behaviour.

    Args:
        nucleus_labels: Integer-labeled nucleus instance mask (H, W).
            0=background, >0=nucleus instance ID.
        dab_channel: DAB concentration map (H, W), float32.
        membrane_mask: Binary membrane mask (H, W), uint8.
        max_cell_radius_px: Maximum distance (pixels) from nucleus edge to
            cell boundary.
        compactness: Watershed compactness parameter. 0=standard watershed.
        distance_sigma: Gaussian sigma for distance transform smoothing.
        min_cell_area_px: Cells smaller than this (in pixels) are removed
            after the watershed step.  Set to 0 to disable (default).
        gradient_weight: Weight of Sobel gradient when blending with DAB
            distance energy (used when membrane mask has gaps).  Range [0, 1].
        dab_weak_threshold: DAB max below this value triggers the pure-gradient
            fallback path.

    Returns:
        Integer-labeled cell instance mask (H, W). Same IDs as nucleus_labels.
    """
    if nucleus_labels.max() == 0:
        return np.zeros_like(nucleus_labels)

    membrane_binary = membrane_mask > 0
    dab_is_weak = _dab_weak(dab_channel, threshold=dab_weak_threshold)

    # --- Build the energy landscape ------------------------------------------
    if dab_is_weak:
        # Path A: DAB signal is too weak -- pure Sobel gradient (Orion fallback)
        gradient = _sobel_gradient_energy(dab_channel)
        # Gradient is high at boundaries -> use directly as energy (watershed
        # finds ridges in the energy image).
        energy = gradient.astype(np.float64)
    else:
        # Path B: DAB is usable -- distance-from-membrane is the primary signal
        distance = ndimage.distance_transform_edt(~membrane_binary).astype(
            np.float64
        )

        if distance_sigma > 0:
            distance = ndimage.gaussian_filter(distance, sigma=distance_sigma)

        # Detect sparse membrane coverage (gaps): if less than 0.5 % of
        # non-nucleus pixels are marked as membrane, blend in the gradient to
        # plug the holes.
        non_nucleus_px = int((nucleus_labels == 0).sum())
        membrane_px = int(membrane_binary.sum())
        membrane_fraction = membrane_px / max(non_nucleus_px, 1)

        if membrane_fraction < 0.005 or gradient_weight > 0:
            gradient = _sobel_gradient_energy(dab_channel)
            # Invert distance so that membranes = high values, then blend.
            dist_max = distance.max()
            if dist_max > 1e-6:
                inv_dist_norm = 1.0 - distance / dist_max
            else:
                inv_dist_norm = np.ones_like(distance)
            w = gradient_weight if membrane_fraction >= 0.005 else max(gradient_weight, 0.5)
            blended = (1.0 - w) * inv_dist_norm + w * gradient.astype(np.float64)
            # Re-invert so that membranes become valleys (low values), suitable
            # for the ``watershed(-energy, ...)`` call below.
            energy = 1.0 - blended
        else:
            energy = distance

    # --- Cap growth at max_cell_radius_px from nucleus edge ------------------
    nucleus_distance = ndimage.distance_transform_edt(nucleus_labels == 0)
    energy[nucleus_distance > max_cell_radius_px] = 0

    # --- Watershed -----------------------------------------------------------
    cell_labels = watershed(
        -energy,
        markers=nucleus_labels,
        mask=energy > 0,
        compactness=compactness,
    )
    cell_labels = cell_labels.astype(np.int32)

    # --- Minimum cell area filtering -----------------------------------------
    if min_cell_area_px > 0:
        for label_id in np.unique(cell_labels):
            if label_id == 0:
                continue
            if int((cell_labels == label_id).sum()) < min_cell_area_px:
                cell_labels[cell_labels == label_id] = 0

    return cell_labels
