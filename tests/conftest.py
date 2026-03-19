"""Shared test fixtures for the instanseg-brightfield test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_tile() -> np.ndarray:
    """A 128x128 synthetic brightfield tile with two 'cells'.

    Creates a purple-blue background (hematoxylin-like) with two circular
    brown regions (DAB-like) that mimic membrane staining around nuclei.
    """
    tile = np.full((128, 128, 3), [200, 180, 220], dtype=np.uint8)  # light purple bg

    # Cell 1: brown membrane ring at (40, 40)
    for y in range(128):
        for x in range(128):
            dist = np.sqrt((x - 40) ** 2 + (y - 40) ** 2)
            if 15 < dist < 20:  # membrane ring
                tile[y, x] = [140, 100, 60]  # brown (DAB)
            elif dist <= 10:  # dark nucleus
                tile[y, x] = [60, 40, 80]  # dark purple

    # Cell 2: brown membrane ring at (90, 80)
    for y in range(128):
        for x in range(128):
            dist = np.sqrt((x - 90) ** 2 + (y - 80) ** 2)
            if 12 < dist < 17:
                tile[y, x] = [150, 110, 70]
            elif dist <= 8:
                tile[y, x] = [50, 30, 70]

    return tile


@pytest.fixture
def hdab_stain_matrix() -> np.ndarray:
    """Standard H-DAB stain matrix (row-normalized)."""
    from instanseg_brightfield.stain import build_stain_matrix
    return build_stain_matrix(
        hematoxylin=[0.650, 0.704, 0.286],
        dab=[0.268, 0.570, 0.776],
        residual=[0.711, 0.424, 0.562],
    )


@pytest.fixture
def nucleus_labels_2cells() -> np.ndarray:
    """Synthetic nucleus label mask with 2 nuclei."""
    labels = np.zeros((128, 128), dtype=np.int32)
    for y in range(128):
        for x in range(128):
            if np.sqrt((x - 40) ** 2 + (y - 40) ** 2) <= 10:
                labels[y, x] = 1
            elif np.sqrt((x - 90) ** 2 + (y - 80) ** 2) <= 8:
                labels[y, x] = 2
    return labels


@pytest.fixture
def cell_labels_2cells() -> np.ndarray:
    """Synthetic cell label mask with 2 cells (larger than nuclei)."""
    labels = np.zeros((128, 128), dtype=np.int32)
    for y in range(128):
        for x in range(128):
            if np.sqrt((x - 40) ** 2 + (y - 40) ** 2) <= 20:
                labels[y, x] = 1
            elif np.sqrt((x - 90) ** 2 + (y - 80) ** 2) <= 17:
                labels[y, x] = 2
    return labels


@pytest.fixture
def sample_dab_channel() -> np.ndarray:
    """Synthetic DAB concentration map with membrane-like rings."""
    dab = np.zeros((128, 128), dtype=np.float32)
    for y in range(128):
        for x in range(128):
            dist1 = np.sqrt((x - 40) ** 2 + (y - 40) ** 2)
            dist2 = np.sqrt((x - 90) ** 2 + (y - 80) ** 2)
            if 15 < dist1 < 20:
                dab[y, x] = 0.5
            if 12 < dist2 < 17:
                dab[y, x] = 0.4
    return dab
