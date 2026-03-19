# Foundation Model Fusion Prototype — Implementation Plan
## Date: 2026-03-19 | Status: Researched, not yet implemented

---

## Overview

Five fusion strategies were identified by 4 research agents (2 Claude + Codex + Gemini) and stress-tested by a 6-agent debate. Debate consensus: **Virchow2 tile filtering (D2) is the only FM strategy worth implementing now.** Scoring fusion is a distraction for IHC — focus on geometry first.

---

## Strategy D2: Virchow2 Epithelial Filtering (APPROVED — Implement First)

### What It Does
Classifies each tile as epithelial vs stromal using Virchow2 foundation model embeddings. Only cells in epithelial tiles are scored for CLDN18.2. This removes stromal cells, lymphocytes, and other non-tumor cells from the scoring denominator.

### Why It Matters
- The VENTANA protocol requires counting only **tumor cells** in the denominator
- Our model detects 5 cells/tile on blue tissue (real cells but NOT epithelial per Virchow2 — 0% epithelial in 8-test validation)
- Filtering these out gives a cleaner, more defensible positivity percentage
- Roche VP debate: "Use a two-model approach — general detector for denominator, DAB model for scoring"

### Implementation Details

**Model:** Virchow2 (Paige AI, 632M params, ViT-H)
- Already cached at `~/.cache/huggingface/hub/models--paige-ai--Virchow2` (2.4 GB)
- Loads via: `timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)`
- Embedding: 2560-dim (CLS 1280 + mean patch 1280)
- VRAM: ~4 GB at batch_size=64 (fits easily on A6000 alongside InstanSeg)

**Classification Approach (from Paige AI CTO debate feedback):**
1. Extract Virchow2 embeddings for all tiles
2. **Critical:** Feed the **hematoxylin channel** (after stain deconvolution), NOT raw IHC — DAB chromogen distorts FM embeddings
3. Use KMeans K=4-5 (NOT K=2) to allow lymphocytes, fibroblasts, endothelial cells their own clusters
4. Validate cluster identity with **morphometrics** (nuclear area, eccentricity, solidity), NOT DAB intensity (avoids circularity — testing DAB selectivity using DAB as ground truth)
5. Assign per-tile epithelial probability
6. Map cells to tiles, set `is_epithelial` flag per cell

**Reference Code:**
- Loading + embeddings: `scripts/score_tiles_foundation_models.py` (lines 240-273)
- KMeans + classification: cldn18 pipeline `scripts/03b_tumor_classifier.py` (lines 253-391)
- Virchow2 utilities: cldn18 pipeline `scripts/virchow2_utils.py` (407 lines)
- Config: diff_undiff `scripts/config.py` (VIRCHOW2_DIM=2560)

**Output:** New Parquet columns: `epithelial_prob` (float64), `is_epithelial` (int8)

### Debate Corrections Applied
| Original Plan | Debate Correction | Source |
|--------------|-------------------|--------|
| K=2 clustering | K=4-5 clustering | Paige AI CTO |
| Raw IHC input | H-channel after deconvolution | Paige AI CTO |
| DAB intensity to validate clusters | Morphometrics (area, eccentricity) | Paige AI CTO |
| Apply to all tiles | Apply only to IHC tiles (not H&E) | Microsoft Research |

---

## Strategy D1: Two-Model Scoring (APPROVED — Implement After D2)

### What It Does
Uses `fluorescence_nuclei_and_cells` model (detects ALL cells regardless of staining, 322 cells/tile on blue tissue) as the denominator, and our model's membrane DAB as the numerator.

### Why
Corrected positivity = (cells with ≥2+ membrane DAB) / (all tumor cells detected by fluorescence model)

This gives the clinically correct fraction that the VENTANA protocol requires.

### Implementation
1. Run fluorescence model on same tiles as our model
2. Match cells by centroid proximity (cKDTree, radius=10px)
3. For matched cells: use our membrane DAB measurement
4. For unmatched cells (detected by fluoro but not our model): score as negative
5. Recompute H-score and positivity % with corrected denominator

**Reference:** fluorescence model at `{venv}/instanseg/bioimageio_models/fluorescence_nuclei_and_cells/0.1.1/instanseg.pt`

---

## Strategy D3: Per-Cell Morphometric Filtering (APPROVED — Simple, No FM Needed)

### What It Does
Filters cells by geometric properties to remove non-epithelial morphologies:
- Nuclear area < 80 μm² → exclude (lymphocytes too small)
- Solidity < 0.9 → exclude (irregular shapes)
- Aspect ratio > 2.0 → exclude (elongated fibroblasts)

### Why
Simple geometric criteria that don't require foundation models. Can be applied immediately to existing Parquet data.

### Implementation
Read polygon_wkb, compute morphometric features via Shapely, filter, recompute scores.

---

## Strategy G3: CellViT++ Integration (DEFERRED — Research Priority Only)

### What It Does
Runs CellViT++ (pip install cellvit) with frozen UNI encoder alongside our model. Extracts per-cell semantic embeddings at zero additional compute during segmentation. Uses embeddings for grade classification refinement.

### Why Deferred
- Debate consensus: focus on geometry first, FM scoring is distraction for IHC
- CellViT++ adds complexity without clear benefit over our membrane measurements
- Best suited for RESEARCH publication showing FM comparison, not for clinical deployment

### Future Implementation
1. `pip install cellvit`
2. Run CellViT++ with UNI encoder on same tiles
3. Extract per-cell embeddings (1280-dim)
4. Train lightweight classifier: [embeddings + our membrane measurements] → grade
5. Compare accuracy vs membrane-only scoring

---

## Strategy B3: CONCH Zero-Shot Membrane Scoring (DEFERRED)

### What It Would Do
Use CONCH vision-language model to compute similarity between tile embeddings and text prompts like "strong membranous staining pattern" vs "absent membranous signal."

### Why Deferred
- Microsoft Research debate: spatial mismatch (FM patch = 112μm, cell = 20μm)
- Gemini debate: foundation models NO-GO for IHC scoring geometry
- Better as independent validation signal than production component

---

## Priority Order

1. **D3** — Morphometric filtering (immediate, no new models needed, 1 hour)
2. **D2** — Virchow2 epithelial filtering (1 day, model cached, code exists)
3. **D1** — Two-model scoring (1 day, both models available)
4. **G3** — CellViT++ (research only, 1 week)
5. **B3** — CONCH zero-shot (research only, deferred)

---

## Validation Results (From 8-Test Suite)

The selectivity validation already tested FM classification:

| Test | Result | Implication |
|------|--------|-------------|
| Test 5 (Virchow2) | 0% of blue-tissue cells classified as epithelial | FM correctly identifies non-epithelial detections |
| Test 6 (Fluorescence) | Fluoro=322 cells on blue, ours=5 | Fluoro model provides true denominator |
| Test 4 (Cross-ref) | 87.8% of blue detections match baseline nuclei | Blue detections are real cells (just not epithelial) |

These results confirm that D2 (Virchow2 filtering) and D1 (two-model scoring) will work as designed.
