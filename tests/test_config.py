"""Tests for configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from instanseg_brightfield.config import (
    DEFAULT_CONFIG,
    get_config_hash,
    load_config,
)


class TestLoadConfig:
    def test_loads_default_config(self):
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "paths" in cfg
        assert "tile_extraction" in cfg
        assert "stain_deconvolution" in cfg

    def test_config_has_all_sections(self):
        cfg = load_config()
        expected_sections = [
            "paths", "tile_extraction", "stain_deconvolution",
            "dab_thresholding", "nucleus_detection", "watershed",
            "quality_filter", "dataset", "training", "evaluation", "export",
        ]
        for section in expected_sections:
            assert section in cfg, f"Missing config section: {section}"

    def test_env_var_expansion(self, monkeypatch):
        monkeypatch.setenv("SLIDE_DIR", "/test/slides")
        cfg = load_config()
        assert cfg["paths"]["slide_dir"] == "/test/slides"

    def test_env_var_default(self, monkeypatch):
        monkeypatch.delenv("SLIDE_DIR", raising=False)
        cfg = load_config()
        assert cfg["paths"]["slide_dir"] == "/pathodata/Claudin18_project/CLDN18_SLIDES_ANON"

    def test_stain_vectors_are_lists(self):
        cfg = load_config()
        assert isinstance(cfg["stain_deconvolution"]["hematoxylin"], list)
        assert len(cfg["stain_deconvolution"]["hematoxylin"]) == 3

    def test_training_config_values(self):
        cfg = load_config()
        assert cfg["training"]["target_segmentation"] == "NC"
        assert cfg["training"]["n_sigma"] == 2
        assert cfg["training"]["dim_coords"] == 2
        assert cfg["training"]["dim_seeds"] == 1


class TestGetConfigHash:
    def test_deterministic(self):
        cfg = load_config()
        h1 = get_config_hash(cfg)
        h2 = get_config_hash(cfg)
        assert h1 == h2

    def test_changes_with_content(self):
        cfg1 = load_config()
        cfg2 = load_config()
        cfg2["training"]["learning_rate"] = 0.999
        assert get_config_hash(cfg1) != get_config_hash(cfg2)

    def test_returns_string(self):
        cfg = load_config()
        h = get_config_hash(cfg)
        assert isinstance(h, str)
        assert len(h) == 12
