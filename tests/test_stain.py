"""Tests for stain deconvolution module."""

from __future__ import annotations

import numpy as np
import pytest

from instanseg_brightfield.stain import build_stain_matrix, deconvolve, extract_dab


class TestBuildStainMatrix:
    def test_row_normalization(self):
        matrix = build_stain_matrix(
            hematoxylin=[0.650, 0.704, 0.286],
            dab=[0.268, 0.570, 0.776],
            residual=[0.711, 0.424, 0.562],
        )
        norms = np.linalg.norm(matrix, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_shape(self):
        matrix = build_stain_matrix([1, 0, 0], [0, 1, 0], [0, 0, 1])
        assert matrix.shape == (3, 3)

    def test_zero_vector_handled(self):
        matrix = build_stain_matrix([0, 0, 0], [0, 1, 0], [0, 0, 1])
        assert not np.any(np.isnan(matrix))


class TestDeconvolve:
    def test_output_shape(self, synthetic_tile, hdab_stain_matrix):
        result = deconvolve(synthetic_tile, hdab_stain_matrix)
        assert result.shape == (128, 128, 3)

    def test_output_dtype(self, synthetic_tile, hdab_stain_matrix):
        result = deconvolve(synthetic_tile, hdab_stain_matrix)
        assert result.dtype == np.float32

    def test_non_negative(self, synthetic_tile, hdab_stain_matrix):
        result = deconvolve(synthetic_tile, hdab_stain_matrix)
        assert np.all(result >= 0)

    def test_white_tile_low_concentration(self, hdab_stain_matrix):
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        result = deconvolve(white, hdab_stain_matrix)
        assert result.max() < 0.01

    def test_dab_channel_higher_at_brown_regions(self, synthetic_tile, hdab_stain_matrix):
        result = deconvolve(synthetic_tile, hdab_stain_matrix)
        dab = result[:, :, 1]
        # Brown membrane ring at radius 15-20 from (40, 40)
        membrane_dab = dab[40, 58]  # at membrane ring
        background_dab = dab[0, 0]  # corner = light purple background
        assert membrane_dab > background_dab


class TestExtractDab:
    def test_returns_2d(self, synthetic_tile, hdab_stain_matrix):
        dab = extract_dab(synthetic_tile, hdab_stain_matrix)
        assert dab.ndim == 2
        assert dab.shape == (128, 128)

    def test_consistent_with_deconvolve(self, synthetic_tile, hdab_stain_matrix):
        dab = extract_dab(synthetic_tile, hdab_stain_matrix)
        full = deconvolve(synthetic_tile, hdab_stain_matrix)
        np.testing.assert_array_equal(dab, full[:, :, 1])
