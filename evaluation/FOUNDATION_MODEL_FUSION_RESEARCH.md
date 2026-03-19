# Foundation Model Fusion Research
## Date: 2026-03-19 | 4 Research Agents (2 Claude + Codex + Gemini)

---

## Executive Summary

Four research agents investigated fusion strategies between our InstanSeg brightfield cell model and pathology foundation models. **Consensus: FMs improve classification (1+ vs 2+) and cell filtering, NOT boundary detection.** The membrane-ring measurement is our unique contribution; FMs add value by selecting which cells to measure, not how to measure them.

---

## Available Foundation Models (Already in Our Codebase)

| Model | Dimensions | Project | Status |
|-------|-----------|---------|--------|
| Virchow2 (Paige AI) | 2560-dim | diff_undiff, cldn18 | Fully integrated |
| CONCH v1.5 (MahmoodLab) | 768-dim | diff_undiff | Fully integrated |
| TITAN (MahmoodLab) | 768-dim slide | diff_undiff | Fully integrated |
| UNI2-h (MahmoodLab) | 1536-dim | cldn18 pipeline | Gated (HF auth) |
| ResNet50 (baseline) | 2048-dim | scoring script | Available |

---

## Strategies Ranked by Feasibility & Impact

### Tier 1: Implement This Week (1 day each)

**Strategy A: Virchow2 Tile Filtering (HIGHEST ROI)**
- Use existing `score_tiles_foundation_models.py` to score all tiles
- Consensus vote: ≥2 of 3 models agree on "epithelial" → keep tile
- Already partially done in codebase
- Expected: +3-7% accuracy, removes stromal/artifact contamination

**Strategy E: Tissue Type Clustering**
- KMeans on Virchow2 embeddings → identify epithelial cluster
- Reuse pattern from cldn18's `03b_tumor_classifier.py`
- Expected: +5-8% H-score accuracy

### Tier 2: Next 2 Weeks

**Strategy B: Post-Segmentation Feature Enrichment (Codex top rec)**
- Run InstanSeg → get cell masks
- Map Virchow2 ViT patch tokens to cells by spatial overlap
- Train XGBoost on [membrane DAB, cell morphology, ViT embedding] → 0/1+/2+/3+
- Expected: Better 1+ vs 2+ discrimination

**Strategy C: CellViT++ Parallel Segmenter (Codex)**
- `pip install cellvit` — frozen UNI encoder + cell segmentation
- Per-cell semantic embeddings at zero extra compute
- Training new classifier: 81 seconds on ~2,500 cells
- Use CellViT++ embeddings + our membrane measurements

### Tier 3: Research-Level (Defer per Debate)

**PCA Feature Augmentation** — FM embeddings as extra input channels
- Microsoft Research: spatial mismatch (ViT 16x downsampling ≠ single-cell resolution)
- Gemini: foundation models NO-GO for IHC scoring geometry

**MIL Cell Filtering** — Gated attention per-cell confidence
- Port from diff_undiff's `32_ensemble_mil.py`
- High research value but 4-7 days implementation

---

## Debate Verdict on FM Fusion

| Agent | Rating |
|-------|--------|
| Microsoft Research | Tile filtering=good, scoring=distraction, PCA=spatial mismatch |
| Gemini | NO-GO for scoring, focus on geometry |
| Codex (discovery) | FMs improve classification, not detection |
| Claude (local) | Strategy A+E first, defer C+D |

**Action:** Implement Strategy A (Virchow2 tile filtering) alongside Track B retraining. Defer all scoring fusion until after pathologist validation.

---

## Key Literature (2024-2025)

| Paper | Year | Relevance |
|-------|------|-----------|
| CellViT++ | Jan 2025 | FM-backed segmentation + per-cell embeddings |
| CellVTA | Apr 2025 | CNN adapter for ViT spatial recovery |
| Cellpose-SAM | May 2025 | SAM encoder + Cellpose flows for IHC |
| CountIHC | 2025 | Rank-aware FM distillation for IHC scoring |
| SMMILe | Nature Cancer 2025 | Spatial MIL quantification |
| IHCFM | IEEE 2024 | First IHC foundation model (347K patches) |
| SegAnyPath | 2024 | Multi-task pathology segmentation FM |
| PathoDuet | MedIA 2024 | H&E to IHC cross-stain transfer |
| ASCO 2025 | 2025 | FMs predict CLDN18.2 from H&E (AUC 0.75) |

---

## Code References

| Strategy | File | What to Reuse |
|----------|------|---------------|
| Tile filtering | `scripts/score_tiles_foundation_models.py` | Virchow2/CONCH/UNI scoring |
| Tissue clustering | cldn18 `scripts/03b_tumor_classifier.py` | KMeans + epithelial detection |
| MIL architecture | diff_undiff `scripts/32_ensemble_mil.py` | GatedAttentionMIL class |
| Ensemble membranes | `scripts/ensemble_membrane_detectors.py` | Fusion strategies |
| Explainability | cldn18 `scripts/12b_foundation_explainability.py` | Attention visualization |
