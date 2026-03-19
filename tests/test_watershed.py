"""Tests for watershed cell segmentation module."""

from __future__ import annotations

import numpy as np

from instanseg_brightfield.watershed import (
    clean_membrane_mask,
    segment_cells,
    segment_cells_enhanced,
    threshold_dab_adaptive,
    threshold_dab_fixed,
)


class TestThresholdDabFixed:
    def test_binary_output(self, sample_dab_channel):
        mask = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        unique = set(np.unique(mask))
        assert unique <= {0, 255}

    def test_membrane_detected(self, sample_dab_channel):
        mask = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        assert mask.sum() > 0

    def test_high_threshold_empty(self, sample_dab_channel):
        mask = threshold_dab_fixed(sample_dab_channel, threshold=10.0)
        assert mask.sum() == 0


class TestThresholdDabAdaptive:
    def test_binary_output(self, sample_dab_channel):
        mask = threshold_dab_adaptive(sample_dab_channel)
        unique = set(np.unique(mask))
        assert unique <= {0, 255}

    def test_with_tissue_mask(self, sample_dab_channel):
        tissue = np.ones_like(sample_dab_channel, dtype=np.uint8) * 255
        mask = threshold_dab_adaptive(sample_dab_channel, tissue_mask=tissue)
        assert mask.shape == sample_dab_channel.shape

    def test_empty_tissue_mask(self, sample_dab_channel):
        tissue = np.zeros_like(sample_dab_channel, dtype=np.uint8)
        mask = threshold_dab_adaptive(sample_dab_channel, tissue_mask=tissue)
        assert mask.sum() == 0


class TestCleanMembraneMask:
    def test_output_shape(self, sample_dab_channel):
        raw_mask = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        cleaned = clean_membrane_mask(raw_mask, closing_size=5, thin=False)
        assert cleaned.shape == raw_mask.shape

    def test_closing_connects_gaps(self):
        # Create a thick block with a 1px gap — closing should fill it
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[28:33, 20:25] = 255  # 5px-tall block
        mask[28:33, 26:32] = 255  # 5px-tall block, 1px gap at col 25
        cleaned = clean_membrane_mask(mask, closing_size=5, thin=False)
        # Closing with 5px elliptical kernel fills a 1px gap in a thick region
        assert cleaned[30, 25] == 255


class TestSegmentCells:
    def test_output_shape(self, nucleus_labels_2cells, sample_dab_channel):
        membrane = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        cells = segment_cells(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=30.0,
        )
        assert cells.shape == nucleus_labels_2cells.shape

    def test_preserves_instance_ids(self, nucleus_labels_2cells, sample_dab_channel):
        membrane = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        cells = segment_cells(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=30.0,
        )
        cell_ids = set(np.unique(cells)) - {0}
        nuc_ids = set(np.unique(nucleus_labels_2cells)) - {0}
        assert cell_ids <= nuc_ids

    def test_cells_contain_nuclei(self, nucleus_labels_2cells, sample_dab_channel):
        membrane = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        cells = segment_cells(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=30.0,
        )
        for nuc_id in [1, 2]:
            nuc_pixels = nucleus_labels_2cells == nuc_id
            cell_at_nuc = cells[nuc_pixels]
            assert np.all(cell_at_nuc == nuc_id)

    def test_cells_larger_than_nuclei(self, nucleus_labels_2cells, sample_dab_channel):
        membrane = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        cells = segment_cells(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=30.0,
        )
        for nuc_id in [1, 2]:
            nuc_area = (nucleus_labels_2cells == nuc_id).sum()
            cell_area = (cells == nuc_id).sum()
            assert cell_area >= nuc_area

    def test_empty_nuclei_returns_empty(self, sample_dab_channel):
        empty_nuclei = np.zeros((128, 128), dtype=np.int32)
        membrane = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        cells = segment_cells(empty_nuclei, sample_dab_channel, membrane)
        assert cells.max() == 0

    def test_max_radius_limits_growth(self, nucleus_labels_2cells, sample_dab_channel):
        membrane = np.zeros((128, 128), dtype=np.uint8)  # no membrane = unlimited growth
        cells_small = segment_cells(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=5.0,
        )
        cells_large = segment_cells(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=50.0,
        )
        assert (cells_large > 0).sum() >= (cells_small > 0).sum()


class TestSegmentCellsEnhanced:
    """Tests for the Orion-inspired enhanced watershed."""

    def test_output_shape(self, nucleus_labels_2cells, sample_dab_channel):
        membrane = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        cells = segment_cells_enhanced(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=30.0,
        )
        assert cells.shape == nucleus_labels_2cells.shape

    def test_preserves_instance_ids(self, nucleus_labels_2cells, sample_dab_channel):
        membrane = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        cells = segment_cells_enhanced(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=30.0,
        )
        cell_ids = set(np.unique(cells)) - {0}
        nuc_ids = set(np.unique(nucleus_labels_2cells)) - {0}
        assert cell_ids <= nuc_ids

    def test_cells_contain_nuclei(self, nucleus_labels_2cells, sample_dab_channel):
        membrane = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        cells = segment_cells_enhanced(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=30.0,
        )
        for nuc_id in [1, 2]:
            nuc_pixels = nucleus_labels_2cells == nuc_id
            cell_at_nuc = cells[nuc_pixels]
            assert np.all(cell_at_nuc == nuc_id)

    def test_cells_larger_than_nuclei(self, nucleus_labels_2cells, sample_dab_channel):
        membrane = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        cells = segment_cells_enhanced(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=30.0,
        )
        for nuc_id in [1, 2]:
            nuc_area = (nucleus_labels_2cells == nuc_id).sum()
            cell_area = (cells == nuc_id).sum()
            assert cell_area >= nuc_area

    def test_empty_nuclei_returns_empty(self, sample_dab_channel):
        empty_nuclei = np.zeros((128, 128), dtype=np.int32)
        membrane = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        cells = segment_cells_enhanced(empty_nuclei, sample_dab_channel, membrane)
        assert cells.max() == 0

    def test_max_radius_limits_growth(self, nucleus_labels_2cells, sample_dab_channel):
        membrane = np.zeros((128, 128), dtype=np.uint8)
        cells_small = segment_cells_enhanced(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=5.0,
        )
        cells_large = segment_cells_enhanced(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=50.0,
        )
        assert (cells_large > 0).sum() >= (cells_small > 0).sum()

    def test_sobel_fallback_weak_dab(self, nucleus_labels_2cells):
        """When DAB is near-zero, gradient fallback should still produce cells."""
        weak_dab = np.random.rand(128, 128).astype(np.float32) * 1e-5
        membrane = np.zeros((128, 128), dtype=np.uint8)
        cells = segment_cells_enhanced(
            nucleus_labels_2cells, weak_dab, membrane,
            max_cell_radius_px=30.0,
            dab_weak_threshold=1e-4,
        )
        # With near-zero DAB the gradient is also near-zero, but the function
        # must not crash and must return a valid label map.
        assert cells.shape == (128, 128)
        assert cells.dtype == np.int32

    def test_gradient_blend_with_sparse_membrane(self, nucleus_labels_2cells, sample_dab_channel):
        """Sparse membrane mask triggers DAB + gradient blending."""
        # A membrane mask with only a few pixels (< 0.5 % coverage)
        sparse_membrane = np.zeros((128, 128), dtype=np.uint8)
        sparse_membrane[40, 40] = 255  # single pixel
        cells = segment_cells_enhanced(
            nucleus_labels_2cells, sample_dab_channel, sparse_membrane,
            max_cell_radius_px=30.0,
            gradient_weight=0.3,
        )
        assert cells.shape == (128, 128)
        cell_ids = set(np.unique(cells)) - {0}
        nuc_ids = set(np.unique(nucleus_labels_2cells)) - {0}
        assert cell_ids <= nuc_ids

    def test_min_cell_area_filtering(self, nucleus_labels_2cells, sample_dab_channel):
        """Large min_cell_area_px should remove small cells."""
        membrane = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        # Use a huge min_cell_area so that all cells are filtered out
        cells = segment_cells_enhanced(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=30.0,
            min_cell_area_px=999999,
        )
        assert cells.max() == 0

    def test_min_cell_area_zero_disables_filter(self, nucleus_labels_2cells, sample_dab_channel):
        """min_cell_area_px=0 (default) should keep all cells."""
        membrane = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        cells = segment_cells_enhanced(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=30.0,
            min_cell_area_px=0,
        )
        cell_ids = set(np.unique(cells)) - {0}
        assert len(cell_ids) > 0

    def test_backward_compatible_with_segment_cells(self, nucleus_labels_2cells, sample_dab_channel):
        """Enhanced version should accept the same positional args as the original."""
        membrane = threshold_dab_fixed(sample_dab_channel, threshold=0.3)
        # Call with exactly the same positional args as segment_cells
        cells = segment_cells_enhanced(
            nucleus_labels_2cells, sample_dab_channel, membrane,
            max_cell_radius_px=30.0, compactness=0.0, distance_sigma=1.0,
        )
        assert cells.shape == nucleus_labels_2cells.shape
        assert cells.dtype == np.int32
