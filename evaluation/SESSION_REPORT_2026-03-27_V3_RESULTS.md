# Session Report: InstanSeg v3.0 + Gen-2 Heatmaps + Label Correction Analysis

**Date**: 2026-03-26 to 2026-03-27
**Duration**: ~12 hours (training + analysis + visualization)

---

## Executive Summary

Trained InstanSeg v3.0 on membrane-aligned masks from Gen-2. The fixed-ring gap improved slightly (-0.020 → -0.016) but still fails — confirming that boundary alignment alone cannot solve the measurement problem. The U-Net pixel classification approach (+0.208) remains the correct solution. Also identified that ~50% of membranes are missed by the model due to bar-filter training label bias. Human annotation correction is the next priority.

---

## InstanSeg v3.0 Retraining

### Setup
- Dataset: 46,119 train / 9,365 val / 9,681 test tiles with membrane-aligned masks
- Aligned masks from Gen-2's Stage 3: watershed re-run with Gen-2's membrane probability as energy landscape
- Training script: 04_train_v2d (sigma clamp schedule, v1.0 weight initialization)
- Hardware: Dual A6000 via DataParallel (InstanSeg's architecture is DP-compatible unlike our attention-gated U-Net)
- Epoch time: ~85 seconds (fast — small model, 4M params)

### Training Progression
| Epoch | test_loss | Sigma clamp | Phase |
|-------|-----------|-------------|-------|
| 0 | 5.401 | -2.0 (tight) | Initial |
| 20 | 4.903 | -2.5 (relaxed) | First big drop — model started producing boundaries |
| 42 | 4.844 | -3.0 | Steady improvement |
| 80 | 4.503 | -3.5 | Strong learning |
| 100 | 4.139 | -4.0 (loose) | Near convergence |
| **130** | **4.069** | -4.0 | **Best checkpoint** |
| 181 | 4.133 | -4.0 | Early stop triggered (patience=50 exhausted) |

**No sigma collapse.** The gradual clamp schedule worked correctly with the new aligned masks.

### Gap Validation
| Model | Fixed-ring gap | Cells | Status |
|-------|---------------|-------|--------|
| v1.0 (original) | -0.020 | 4,124 | FAIL |
| **v3.0 (aligned)** | **-0.016** | **1,562** | **FAIL (slightly improved)** |
| Bar-filter | +0.127 | 6,626 | PASS |
| Gen-1 U-Net | +0.208 | 4,233 | PASS |
| Gen-2 U-Net | +0.209 | 4,307 | PASS |

### Analysis
The fixed-ring gap improved by 0.004 (from -0.020 to -0.016) — the boundary DID move closer to the membrane. But the improvement is marginal because:
1. The fixed-ring measurement geometry averages across the membrane/cytoplasm boundary regardless of boundary placement
2. Cell count dropped 62% (4,124 → 1,562) — v3.0 detects fewer cells
3. Seed confidence is lower (cell_seed max = 0.93 vs v1.0's ~0.99)

**Conclusion:** Boundary alignment helps slightly but the fixed-ring measurement is fundamentally limited. The U-Net pixel classification remains the correct measurement approach.

### Technical Issue
The training checkpoint failed to save on the final epoch (relative path `../models/` didn't resolve correctly). However, intermediate best-epoch checkpoints were saved successfully at:
`/home/fernandosoto/Documents/models/brightfield_cells_nuclei_v2.1_gradualClamp/brightfield_cells_nuclei_v2.1_gradualClamp/model_weights.pth`

Key needed: `module.` prefix stripping from DataParallel keys when loading.

---

## Gen-2 Heatmap Analysis

### Coloring-Book Heatmaps (Pixel-Level Semantic Overlay)
Generated per-tile overlays using Gen-2's 4-class predictions:
- Membrane: red/warm colors scaled by DAB intensity (using RdYlBu_r colormap)
- Nucleus: blue
- Cytoplasm: light teal
- Background: transparent

### Critical Finding: ~50% Membrane Recall
When visually inspecting tiles, approximately 50% of visible brown membranes are NOT detected by the model. This was observed across all staining intensities (strong, moderate, weak).

**Root cause:** The bar-filter training labels missed those membranes.
- Bar-filter detects THIN, LINEAR DAB structures
- Thick membranes (wide DAB band) → bar-filter sees as blob, not line
- Diffuse membranes (gradual DAB transition) → no sharp edge for bar-filter
- The model faithfully learned the bar-filter's blind spots

**User observation (critical):** "It starts very well but then suddenly stops. There is no reason why it should not keep catching membranes, the other membranes it missed are just not so different." — This confirms the bar-filter label inconsistency hypothesis. Adjacent identical glands are labeled differently because the bar-filter's local response varies with background contrast.

### Attempted Fixes
1. **Argmax vs soft threshold vs continuous probability** — Continuous was slightly better but still missed ~50%
2. **Hybrid (model + raw DAB)** — Too aggressive, bled into cytoplasm
3. **DAB-at-cell-periphery criterion** — Only added 0.2-1.0% more membrane (cell masks too small)
4. **Auto-correction using detected membrane statistics** — Only 0.8% improvement (too conservative)

**Conclusion:** Algorithmic fixes cannot solve this. The model needs human-corrected training labels.

---

## Annotation Correction Setup

Exported 60 tiles for manual annotation correction:
- Location: `evaluation/annotation_correction/`
- Sorted by staining intensity (strongest first)
- Each folder contains:
  - `original.png` — raw tissue
  - `model_prediction.png` — Gen-2 overlay (red = detected membrane)
  - `correction_template.png` — color-coded label for editing
  - `PAINT_HERE.png` — tissue with red tint on detected membrane (paint bright red where missing)

### Workflow for Gen-4
1. User annotates 20-50 tiles (paint red where membrane is missing)
2. Merge corrections with bar-filter labels
3. Retrain Gen-3/Gen-4 on corrected labels
4. Optionally combine with multi-vector training for scanner robustness

---

## Multi-Site Staining Discovery (Updated)

Confirmed: slides come from ~20 different hospitals, all scanned at MUHC on Hamamatsu NanoZoomer.

This means:
- Multi-site STAINING variation already in training data
- Single-site SCANNING (Hamamatsu only)
- The model has been exposed to diverse staining protocols
- Scanner robustness (Leica Aperio AT2) is the only untested axis

---

## Dilution Slides for Calibration

User confirmed availability of:
- Multiple dilution slides (non-diluted to extremely diluted)
- TMAs with known CLDN18.2 status
- All from the same lab and machines

These are critical for:
1. Threshold calibration (standard curve across dilutions)
2. Analytical sensitivity (limit of detection)
3. Linearity testing (does measurement scale with dilution?)
4. Resolving the 82% at 3+ question

---

## Curriculum Update

7 notebooks pushed to DrSoma/pathology-notebooks (5,511 lines):
- MC_014: Oriented Gaussian bar-filters (+15 cells)
- MC_017: Sigma collapse debugging (+11 cells)
- AC_08: Attention gates + embedding segmentation (+7 cells)
- MC_020: Classical-to-Neural Knowledge Distillation (new, 25 cells)
- MC_021: Membrane Biophysics Measurement (new, 23 cells)
- MC_022: GPU Optimization for Production (new, 32 cells)
- MC_023: Multi-Agent AI Review (new, 21 cells)

---

## Model Generation Summary

| Generation | Architecture | Trained On | Gap | mem% | Purpose |
|-----------|-------------|------------|-----|------|---------|
| Bar-filter | Classical (Aperio) | N/A | +0.127 | N/A | Original teacher |
| Gen-1 | UNet 32/64/128/256 (7.8M) | Bar-filter labels | +0.208 | 71% | Breakthrough — surpassed teacher |
| Gen-2 | WiderAttnUNet 48/96/192/384 (17.7M) | Gen-1 soft labels | +0.209 | 90% | Better pixel classification, same measurement |
| InstanSeg v3.0 | InstanSeg UNet (4M) | Aligned masks | -0.016 | N/A | Boundary closer but ring still fails |
| **Gen-4 (next)** | **TBD** | **Human-corrected labels** | **TBD** | **TBD** | **Fix 50% miss rate** |

---

## Next Steps (Priority Order)

1. **Human annotation of 20-50 tiles** — fix bar-filter's missed membranes
2. **Gen-4 on corrected labels** — potentially with multi-vector for robustness
3. **Dilution slide validation** — threshold calibration + linearity
4. **Full-slide Gen-2 heatmaps** — currently generating CLDN0042
5. **Pathologist calibration** — Dr. Fiset scores 30-tile export
6. **McGill tech transfer** — protect IP
7. **100-cell spot check** — physician annotation for paper

---

## Hardware Notes
- DataParallel works for InstanSeg (no attention gates) — dual GPU at ~85 sec/epoch
- DataParallel FAILS for WiderAttnUNet (F.interpolate in attention gates causes CUDA error)
- InstanSeg training saves checkpoints with `module.` prefix — must strip when loading
- `../models/` relative path fails when CWD is different — always use absolute paths
- Restored original dataset symlink after v3.0 training: `data/segmentation_dataset.pth` → original
