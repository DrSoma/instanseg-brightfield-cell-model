"""Tests for pipeline state management."""

from __future__ import annotations

from pathlib import Path

from instanseg_brightfield.pipeline_state import PipelineManifest


class TestPipelineManifest:
    def test_create_new(self, tmp_path):
        manifest = PipelineManifest(tmp_path / "manifest.json", "abc123", "test_step")
        assert manifest.completed_count == 0

    def test_mark_and_check(self, tmp_path):
        manifest = PipelineManifest(tmp_path / "manifest.json", "abc123")
        assert not manifest.is_complete("tile_001")
        manifest.mark_complete("tile_001", stats={"cells": 5})
        assert manifest.is_complete("tile_001")
        assert manifest.completed_count == 1

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "manifest.json"
        m1 = PipelineManifest(path, "abc123")
        m1.mark_complete("tile_001")
        m1.mark_complete("tile_002")
        m1.save()

        m2 = PipelineManifest(path, "abc123")
        assert m2.is_complete("tile_001")
        assert m2.is_complete("tile_002")
        assert m2.completed_count == 2

    def test_config_change_invalidates(self, tmp_path):
        path = tmp_path / "manifest.json"
        m1 = PipelineManifest(path, "abc123")
        m1.mark_complete("tile_001")
        m1.save()

        m2 = PipelineManifest(path, "def456")  # different config hash
        assert not m2.is_complete("tile_001")
        assert m2.completed_count == 0

    def test_get_all_stats(self, tmp_path):
        manifest = PipelineManifest(tmp_path / "manifest.json", "abc123")
        manifest.mark_complete("t1", stats={"cells": 5})
        manifest.mark_complete("t2", stats={"cells": 3})
        manifest.mark_complete("t3")  # no stats

        all_stats = manifest.get_all_stats()
        assert len(all_stats) == 2
        assert all_stats[0]["cells"] == 5
