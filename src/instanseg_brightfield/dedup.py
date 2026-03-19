"""Hybrid deduplication for overlapping tiles.

Ported from Orion's wholeslide dedup (orion_ml_worker/wholeslide/dedup.py).
Center cells (centroid in the non-overlap zone) are always kept.  Padding
cells (centroid in the overlap zone) are only kept when no center cell from
an adjacent tile falls within *dedup_radius*.

A spatial grid (bucket_size = 2 * radius) gives O(1) neighbour lookup.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def deduplicate_cells(
    cells: list[dict],
    dedup_radius: float,
) -> list[dict]:
    """Hybrid deduplication: keep all center cells, add orphan padding cells.

    Parameters
    ----------
    cells : list[dict]
        Each dict must contain at minimum:
          - ``centroid_x`` (float): global x coordinate of the cell centroid.
          - ``centroid_y`` (float): global y coordinate of the cell centroid.
          - ``is_center`` (bool): True if the centroid falls inside the
            non-overlap (content) zone of its source tile.
    dedup_radius : float
        Maximum distance (in pixels) at which a padding cell is considered
        a duplicate of a center cell.

    Returns
    -------
    list[dict]
        Deduplicated cell list.  All center cells are retained.  A padding
        cell is kept only when no center cell exists within *dedup_radius*.
    """
    center_cells = [c for c in cells if c.get("is_center", True)]
    padding_cells = [c for c in cells if not c.get("is_center", True)]

    if not padding_cells:
        return list(center_cells)

    # Build spatial grid from center cells for O(1) neighbour lookup
    bucket_size = max(1, int(2 * dedup_radius))
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, c in enumerate(center_cells):
        gx = int(c["centroid_x"]) // bucket_size
        gy = int(c["centroid_y"]) // bucket_size
        grid[(gx, gy)].append(i)

    r2 = dedup_radius ** 2
    orphans: list[dict] = []

    for pc in padding_cells:
        px, py = pc["centroid_x"], pc["centroid_y"]
        gx = int(px) // bucket_size
        gy = int(py) // bucket_size

        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for ci in grid.get((gx + dx, gy + dy), ()):
                    cc = center_cells[ci]
                    dist2 = (px - cc["centroid_x"]) ** 2 + (py - cc["centroid_y"]) ** 2
                    if dist2 <= r2:
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if not found:
            orphans.append(pc)

    result = center_cells + orphans
    logger.info(
        "Dedup: %d center + %d orphan padding = %d kept (from %d raw)",
        len(center_cells),
        len(orphans),
        len(result),
        len(cells),
    )
    return result
