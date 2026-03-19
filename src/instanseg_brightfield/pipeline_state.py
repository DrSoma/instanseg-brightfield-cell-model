"""Pipeline state management for idempotent, resumable execution.

Each pipeline step records completion status per-tile in a JSON manifest.
If a step is interrupted, re-running it skips already-completed tiles.
Manifests also store provenance: config hash, git commit, per-tile stats.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def get_git_sha() -> str:
    """Get the current git commit SHA, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


class PipelineManifest:
    """Tracks per-tile completion for a pipeline step.

    Usage:
        manifest = PipelineManifest(data_dir / "tiles" / "manifest.json", config_hash)
        for tile_id in tile_ids:
            if manifest.is_complete(tile_id):
                continue
            # ... process tile ...
            manifest.mark_complete(tile_id, stats={"num_cells": 42})
        manifest.save()
    """

    def __init__(self, path: Path, config_hash: str, step_name: str = ""):
        self.path = Path(path)
        self.config_hash = config_hash
        self.step_name = step_name
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            if data.get("config_hash") != self.config_hash:
                # Config changed — invalidate previous results
                return self._new_manifest()
            return data
        return self._new_manifest()

    def _new_manifest(self) -> dict:
        return {
            "step_name": self.step_name,
            "config_hash": self.config_hash,
            "git_sha": get_git_sha(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tiles": {},
        }

    def is_complete(self, tile_id: str) -> bool:
        return tile_id in self._data["tiles"]

    def mark_complete(self, tile_id: str, stats: dict | None = None):
        entry = {"completed_at": datetime.now(timezone.utc).isoformat()}
        if stats:
            entry["stats"] = stats
        self._data["tiles"][tile_id] = entry

    def save(self):
        import os as _os
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
            f.flush()
            _os.fsync(f.fileno())
        tmp.rename(self.path)

    @property
    def completed_count(self) -> int:
        return len(self._data["tiles"])

    def get_all_stats(self) -> list[dict]:
        """Collect stats from all completed tiles for distribution monitoring."""
        return [
            entry["stats"]
            for entry in self._data["tiles"].values()
            if "stats" in entry
        ]
