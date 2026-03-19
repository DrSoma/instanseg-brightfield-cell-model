"""Stain deconvolution using Beer-Lambert law.

Separates H-DAB brightfield images into Hematoxylin, DAB, and Residual channels.
Matches the implementation in Orion's processor.py and QuPath's color deconvolution.
"""

from __future__ import annotations

import numpy as np
import torch


def build_stain_matrix(
    hematoxylin: list[float],
    dab: list[float],
    residual: list[float],
) -> np.ndarray:
    """Build a row-normalized 3x3 stain matrix.

    Args:
        hematoxylin: RGB optical density vector for hematoxylin.
        dab: RGB optical density vector for DAB.
        residual: RGB optical density vector for residual channel.

    Returns:
        Row-normalized 3x3 stain matrix (float64).
    """
    matrix = np.array([hematoxylin, dab, residual], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def deconvolve(
    tile_rgb: np.ndarray,
    stain_matrix: np.ndarray,
) -> np.ndarray:
    """Deconvolve an RGB tile into stain concentration channels.

    Uses the Beer-Lambert law: OD = -log(I/I0), then solves for concentrations
    by inverting the stain matrix.

    Args:
        tile_rgb: RGB image, uint8 (H, W, 3).
        stain_matrix: Row-normalized 3x3 stain matrix.

    Returns:
        Stain concentrations (H, W, 3) float32.
        Channel 0 = Hematoxylin, Channel 1 = DAB, Channel 2 = Residual.
    """
    rgb_float = np.clip(tile_rgb.astype(np.float64), 1, 255) / 255.0
    optical_density = -np.log(rgb_float)
    inv_matrix = np.linalg.inv(stain_matrix)
    h, w = tile_rgb.shape[:2]
    od_flat = optical_density.reshape(-1, 3)
    concentrations_flat = od_flat @ inv_matrix
    concentrations = concentrations_flat.reshape(h, w, 3)
    return np.clip(concentrations, 0, None).astype(np.float32)


def extract_dab(
    tile_rgb: np.ndarray,
    stain_matrix: np.ndarray,
) -> np.ndarray:
    """Extract the DAB concentration channel from an RGB tile.

    Args:
        tile_rgb: RGB image, uint8 (H, W, 3).
        stain_matrix: Row-normalized 3x3 stain matrix.

    Returns:
        DAB concentration map (H, W) float32, values >= 0.
    """
    return deconvolve(tile_rgb, stain_matrix)[:, :, 1]


# ---------------------------------------------------------------------------
# Class-based interface: precompute the inverse matrix once, reuse per tile
# ---------------------------------------------------------------------------


class StainDeconvolver:
    """Stain deconvolver that precomputes the inverse stain matrix once.

    The standalone :func:`deconvolve` function calls ``np.linalg.inv`` on every
    invocation.  This class computes the row-normalized inverse matrix in
    ``__init__`` and reuses it for every subsequent tile, which is faster when
    processing many tiles with the same stain vectors.

    It also exposes a GPU path (:meth:`deconvolve_gpu`) that runs the same
    Beer-Lambert math on a CUDA tensor so that stain separation can stay on
    device without a round-trip to NumPy.

    Args:
        stain_matrix: Un-normalized or pre-normalized 3x3 stain matrix
            (rows = stain vectors, columns = RGB optical densities).
    """

    def __init__(self, stain_matrix: np.ndarray) -> None:
        stain_matrix = np.asarray(stain_matrix, dtype=np.float64)
        norms = np.linalg.norm(stain_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        norm = stain_matrix / norms
        self._inv: np.ndarray = np.linalg.inv(norm)  # computed ONCE
        self._inv_tensor: torch.Tensor | None = None  # lazy GPU init

    # -- CPU numpy path -----------------------------------------------------

    def deconvolve(self, tile_rgb: np.ndarray) -> np.ndarray:
        """Deconvolve an RGB tile into stain concentrations (CPU/NumPy).

        Args:
            tile_rgb: RGB image, uint8 (H, W, 3).

        Returns:
            Stain concentrations (H, W, 3) float32.
            Channel 0 = Hematoxylin, 1 = DAB, 2 = Residual.
        """
        od = -np.log(np.clip(tile_rgb.astype(np.float64) / 255.0, 1 / 255, 1))
        return np.clip(od @ self._inv, 0, None).astype(np.float32)

    def extract_dab(self, tile_rgb: np.ndarray) -> np.ndarray:
        """Extract DAB concentration channel (CPU/NumPy).

        Args:
            tile_rgb: RGB image, uint8 (H, W, 3).

        Returns:
            DAB concentration map (H, W) float32, values >= 0.
        """
        return self.deconvolve(tile_rgb)[:, :, 1]

    # -- GPU tensor path ----------------------------------------------------

    def deconvolve_gpu(
        self,
        batch_tensor: torch.Tensor,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Deconvolve a batch of RGB tiles on GPU.

        Args:
            batch_tensor: Float32 tensor in ``[0, 1]`` range.
                Accepts **(B, 3, H, W)** or **(3, H, W)**.
            device: Target CUDA (or CPU) device.

        Returns:
            Stain concentrations with the same shape as *batch_tensor*
            (``(B, 3, H, W)`` or ``(3, H, W)``).
        """
        if self._inv_tensor is None or self._inv_tensor.device != torch.device(device):
            self._inv_tensor = torch.from_numpy(
                self._inv.astype(np.float32)
            ).to(device)

        unbatched = batch_tensor.ndim == 3
        if unbatched:
            batch_tensor = batch_tensor.unsqueeze(0)

        B, C, H, W = batch_tensor.shape
        # (B, 3, H, W) -> (B*H*W, 3)
        pixels = batch_tensor.permute(0, 2, 3, 1).reshape(-1, 3)
        od = -torch.log(pixels.clamp(min=1 / 255))
        conc = (od @ self._inv_tensor).clamp(min=0)
        result = conc.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)

        if unbatched:
            result = result.squeeze(0)
        return result

    def extract_dab_gpu(
        self,
        batch_tensor: torch.Tensor,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Extract DAB concentration channel on GPU.

        Args:
            batch_tensor: Float32 tensor in ``[0, 1]`` range.
                Accepts **(B, 3, H, W)** or **(3, H, W)**.
            device: Target CUDA (or CPU) device.

        Returns:
            DAB map — **(B, H, W)** or **(H, W)** depending on input shape.
        """
        deconv = self.deconvolve_gpu(batch_tensor, device)
        if deconv.ndim == 3:
            return deconv[1]  # (H, W)
        return deconv[:, 1]  # (B, H, W)
