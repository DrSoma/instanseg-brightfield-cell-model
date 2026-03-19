"""Tests for quality filtering module."""

from __future__ import annotations

import numpy as np

from instanseg_brightfield.quality import compute_tile_stats, filter_cells


class TestFilterCells:
    def test_keeps_valid_cells(
        self, nucleus_labels_2cells, cell_labels_2cells, sample_dab_channel,
    ):
        filt_nuc, filt_cell, stats = filter_cells(
            nucleus_labels_2cells, cell_labels_2cells, sample_dab_channel,
            max_cell_nucleus_ratio=10.0,
            min_membrane_coverage=0.0,  # disable membrane check
            min_nucleus_area_px=1,
            max_nucleus_area_px=10000,
            min_cell_area_px=1,
        )
        assert stats["kept_instances"] > 0
        assert filt_nuc.max() > 0
        assert filt_cell.max() > 0

    def test_filters_small_nuclei(
        self, nucleus_labels_2cells, cell_labels_2cells, sample_dab_channel,
    ):
        _, _, stats = filter_cells(
            nucleus_labels_2cells, cell_labels_2cells, sample_dab_channel,
            min_nucleus_area_px=100000,  # impossibly large
            min_membrane_coverage=0.0,
        )
        assert stats["kept_instances"] == 0
        assert stats["removed_reasons"]["small_nucleus"] > 0

    def test_filters_high_ratio(
        self, nucleus_labels_2cells, cell_labels_2cells, sample_dab_channel,
    ):
        _, _, stats = filter_cells(
            nucleus_labels_2cells, cell_labels_2cells, sample_dab_channel,
            max_cell_nucleus_ratio=1.0,  # impossibly strict
            min_membrane_coverage=0.0,
            min_nucleus_area_px=1,
        )
        # Cells are larger than nuclei so ratio > 1.0
        assert stats["removed_reasons"]["high_ratio"] > 0

    def test_contiguous_ids(
        self, nucleus_labels_2cells, cell_labels_2cells, sample_dab_channel,
    ):
        filt_nuc, filt_cell, _ = filter_cells(
            nucleus_labels_2cells, cell_labels_2cells, sample_dab_channel,
            min_membrane_coverage=0.0,
            min_nucleus_area_px=1,
            max_nucleus_area_px=10000,
            min_cell_area_px=1,
            max_cell_nucleus_ratio=10.0,
        )
        nuc_ids = sorted(set(np.unique(filt_nuc)) - {0})
        if nuc_ids:
            assert nuc_ids == list(range(1, len(nuc_ids) + 1))

    def test_stats_dict_structure(
        self, nucleus_labels_2cells, cell_labels_2cells, sample_dab_channel,
    ):
        _, _, stats = filter_cells(
            nucleus_labels_2cells, cell_labels_2cells, sample_dab_channel,
            min_membrane_coverage=0.0,
            min_nucleus_area_px=1,
        )
        assert "total_instances" in stats
        assert "kept_instances" in stats
        assert "removed_reasons" in stats


class TestComputeTileStats:
    def test_returns_expected_keys(self, nucleus_labels_2cells, cell_labels_2cells):
        stats = compute_tile_stats(nucleus_labels_2cells, cell_labels_2cells)
        expected_keys = {
            "num_nuclei", "num_cells", "mean_nucleus_area", "mean_cell_area",
            "mean_cell_nucleus_ratio", "std_cell_nucleus_ratio", "tile_coverage",
        }
        assert set(stats.keys()) == expected_keys

    def test_correct_counts(self, nucleus_labels_2cells, cell_labels_2cells):
        stats = compute_tile_stats(nucleus_labels_2cells, cell_labels_2cells)
        assert stats["num_nuclei"] == 2
        assert stats["num_cells"] == 2

    def test_empty_masks(self):
        empty = np.zeros((64, 64), dtype=np.int32)
        stats = compute_tile_stats(empty, empty)
        assert stats["num_nuclei"] == 0
        assert stats["num_cells"] == 0
        assert stats["tile_coverage"] == 0.0
