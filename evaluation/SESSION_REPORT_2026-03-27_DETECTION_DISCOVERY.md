# Session Report: Root Cause Discovery — Nucleus Detection, Not Bar-Filter

**Date**: 2026-03-27
**Significance**: Identified the true root cause of the 50% membrane miss rate

---

## Executive Summary

Through systematic investigation, discovered that the membrane detection problem has TWO root causes, not one:

1. **Nucleus detection gap**: Our fine-tuned InstanSeg detects only 90 nuclei per tile vs 266 by the stock model — a 66% loss caused by training data filtered to DAB-positive cells only. Cells that aren't detected can't have their membrane measured.

2. **Gen-2 training label gap**: Even for detected cells, Gen-2's training labels (from bar-filter) are incomplete. Gen-2 was trained on data from 90 cells/tile, so it learned "background" for the regions where the other 302 cells exist.

The solution: regenerate training labels using a 4-model ensemble for nucleus detection (392 nuclei/tile), then retrain Gen-3 on the expanded labels.

---

## Discovery Timeline

### Starting Point: "The Model Stops Detecting Membranes Suddenly"

User observed that on heatmaps, the model detects membrane perfectly on some glands but completely ignores adjacent identical glands. There's no biological reason for this cutoff.

### Hypothesis 1: Bar-Filter Label Inconsistency (PARTIALLY CORRECT)

Initial diagnosis: the bar-filter's response varies with local background contrast, producing inconsistent labels. Some glands get labeled as membrane, adjacent identical glands don't.

**What we tried:**
- Argmax vs soft threshold vs continuous probability → continuous slightly better but still ~50% miss
- Hybrid model + raw DAB → too aggressive, bled into cytoplasm
- DAB-at-cell-periphery criterion → only +0.2-1.0% more membrane
- Auto-correction using detected membrane statistics → only +0.8%

**Conclusion:** Bar-filter inconsistency is real but isn't the main problem.

### Hypothesis 2: Quality Filter Too Aggressive (INCORRECT)

Tested lowering `min_membrane_coverage` from 0.3 to 0.0:
- threshold=0.30: 191 cells survive
- threshold=0.00: 191 cells survive (SAME)

The membrane coverage filter isn't the bottleneck. Other quality filters (area, solidity, cell-nucleus ratio) remove the same 71 cells regardless.

### Hypothesis 3: Nucleus Detection (THE ROOT CAUSE)

User asked: "What if the issue is the nuclei segmentation model itself not recognizing all of the cells?"

**Three-model comparison on the same tile:**

| Model | Nuclei Detected |
|-------|----------------|
| Stock brightfield_nuclei | 266 |
| Our fine-tuned model | 90 |
| Fluorescence model | 175 |

**Our fine-tuned model detects 66% fewer nuclei than the stock model.**

### Why Our Model Detects Fewer Nuclei

During training data generation (script 02), the quality filter `min_membrane_coverage=0.3` removed cells without strong DAB membrane staining. Our model was trained on this filtered dataset and learned: "a cell looks like a nucleus surrounded by brown membrane." It stopped recognizing nuclei without visible membrane — including many epithelial cells with moderate/weak staining.

The stock `brightfield_nuclei` model was never exposed to this filter. It detects ALL blue nuclei regardless of DAB.

### Critical Bias in HTML Report

User caught that the comparison in the selectivity validation HTML report (327,641 baseline vs 41,535 ours) was biased: our model had the tissue mask applied (counting only within epithelial regions), while the baseline counted ALL cells across the entire slide. The real gap for epithelial cells is much smaller.

User also observed from the hybrid heatmap that "the amount of cells it targets on epithelium is the same" — the stock model's extra detections were mostly in stroma, not epithelium.

### Why Fine-Tuning Was Necessary (Bootstrap)

User asked: "Why did we fine-tune in the first place?"

Answer: To get **cell boundaries** (not just nuclei). The stock model only detects nuclei — no cell head. We needed dual-head output (nuclei + cells) to define membrane rings for the bar-filter measurement.

**But now Gen-2 U-Net replaces that function.** Gen-2 classifies every pixel as membrane/cytoplasm/nucleus/background — it doesn't need InstanSeg's cell boundaries. The fine-tuned model served as a bootstrap to create Gen-2, but the production pipeline doesn't need it anymore.

---

## Pipeline Comparison (Definitive)

| Pipeline | Gap | Cells | Notes |
|----------|-----|-------|-------|
| A: Our fine-tuned + Gen-2 | +0.209 | 4,301 | Current pipeline |
| B: Stock nuclei + Gen-2 | +0.221 | 7,244 | Better detection, better gap |
| C: Stock + fixed ring (no Gen-2) | +0.006 | 22,344 | Proves Gen-2 is essential |
| **D: Ensemble (stock+fluoro) + Gen-2** | **+0.236** | **7,443** | **Best gap** |

**Pipeline D wins.** But the additional cells from the ensemble don't have membrane overlay because Gen-2 wasn't trained on them.

---

## Complementary Detection: Why Models See Different Nuclei

**Stock brightfield_nuclei**: Trained on H&E/IHC. Detects nuclei by looking for blue/purple blobs. Misses nuclei where heavy brown DAB covers the hematoxylin signal.

**Fluorescence model**: Trained on DAPI fluorescence. Detects nuclei by texture/density patterns, not blue color specifically. Finds nuclei under heavy DAB where brightfield model fails. Misses some nuclei that need brightfield-specific contrast.

**Why they complement each other**: Different training data → different learned features → different failure modes → union covers more cells.

---

## Hematoxylin Deconvolution Trick

Since nuclei are missed because DAB hides hematoxylin, we computationally removed DAB via stain deconvolution and ran nucleus detection on the hematoxylin-only image:

| Run | Image | Nuclei |
|-----|-------|--------|
| Stock | Original | 266 |
| Fluoro | Original | 175 |
| Stock | Hematoxylin-only | 284 (+18 new) |
| Fluoro | Hematoxylin-only | 86 (worse — needs color contrast) |
| **Full ensemble (4 runs)** | **Combined** | **392** |

The stock model found 18 additional nuclei on the hematoxylin-only image that were hidden under DAB in the original. Full 4-run ensemble: **392 nuclei (4.4x our fine-tuned model's 90)**.

---

## The Remaining Gap: Gen-2 Training Labels

Even with 392 nuclei detected by the ensemble, Gen-2's membrane overlay doesn't cover the new cells because Gen-2 was trained on labels generated from only 90 cells/tile. The 302 additional cells were labeled as "background" in Gen-2's training data.

**The fix:** Regenerate bar-filter training labels using the ensemble's 392 nuclei instead of our model's 90. This would give Gen-3 4x more membrane examples to learn from. The cells that Gen-2 currently ignores would become training examples with proper membrane labels.

---

## Final Architecture (Proposed)

```
DETECTION (4-run ensemble, ~60ms):
  Stock brightfield_nuclei on original image → 266 nuclei
  Fluorescence model on original image → 175 nuclei
  Stock brightfield_nuclei on hematoxylin-only → 284 nuclei
  Fluorescence on hematoxylin-only → 86 nuclei
  Union → 392 nuclei

CLASSIFICATION (Gen-2 U-Net, ~2ms):
  RGB tile → 4-class pixel classification (bg/nuc/cyto/membrane)

MEASUREMENT:
  For each detected nucleus → expand to cell region
  Measure DAB in Gen-2's predicted membrane pixels within each cell
  Score 0/1+/2+/3+ per cell
  Compute H-score and eligibility
```

---

## Key Insights from This Session

### 1. The Fine-Tuned Model Was a Necessary Bootstrap, Not the Final Product
It created the training data chain: fine-tuned cells → bar-filter labels → Gen-1 → Gen-2. Now that Gen-2 exists, the stock model is the better nucleus detector.

### 2. Quality Filters at Training Time Propagate as Permanent Bias
`min_membrane_coverage=0.3` during mask generation permanently excluded 66% of cells from the training pipeline. Every downstream model inherited this blind spot. Future training should use minimal or no quality filtering.

### 3. Model Ensembles Are More Effective Than Fine-Tuning for Detection
Fine-tuning one model for selective detection lost 66% of cells. Ensembling two stock models with a deconvolution trick gained 47% more cells. Different failure modes + union > specialized single model.

### 4. The Measurement and Detection Problems Are Separate
Gen-2 solves measurement (gap +0.209). The ensemble solves detection (392 vs 90 nuclei). Combining them gives the best of both worlds, but Gen-2 needs retraining on the expanded labels to classify membrane around the newly detected cells.

### 5. Bias in Validation Can Mislead Conclusions
The HTML report comparison (327k vs 42k) was unfair because one had a tissue mask and the other didn't. Always apply identical preprocessing to both arms of a comparison.

---

## Next Steps

1. **Regenerate bar-filter labels using the ensemble's nuclei** (392/tile instead of 90)
2. **Retrain Gen-3 on expanded labels** — Gen-3 will see membrane in 4x more cells
3. **Run gap validation on the new pipeline** (ensemble + Gen-3)
4. **Generate full-slide heatmaps** with complete coverage
5. **Dilution slide validation** for threshold calibration
6. **Pathologist calibration** (Dr. Fiset)
7. **Document and publish**

---

## Files Created This Session

| File | Purpose |
|------|---------|
| `evaluation/coloring_book_heatmaps/nucleus_model_comparison.png` | 4-panel: original vs stock vs ours vs fluorescence nuclei |
| `evaluation/coloring_book_heatmaps/ensemble_nuclei_comparison.png` | 5-panel: stock + fluoro + ensemble + highlight |
| `evaluation/coloring_book_heatmaps/membrane_threshold_comparison.png` | Quality filter threshold test |
| `evaluation/coloring_book_heatmaps/hybrid_pipeline_comparison.png` | Old pipeline vs stock+Gen2 |
| `evaluation/coloring_book_heatmaps/hybrid_vs_old_strong.png` | Strong tile: 51 → 346 cells |
| `evaluation/coloring_book_heatmaps/hybrid_vs_old_moderate.png` | Moderate tile: 109 → 345 cells |
| `evaluation/coloring_book_heatmaps/hematoxylin_deconv_detection.png` | 7-panel: deconvolution trick results |
| `evaluation/coloring_book_heatmaps/full_ensemble_pixel_heatmap.png` | Full ensemble + Gen-2 membrane overlay |
| `evaluation/coloring_book_heatmaps/CLDN0042_gen2_fullslide.png` | Full slide Gen-2 semantic heatmap (16384px) |
| `evaluation/coloring_book_heatmaps/CLDN0042_gen2_fullslide_crop.png` | Cropped version |
| `evaluation/instanseg_v3_gap_results.json` | InstanSeg v3.0 gap results |
| `evaluation/SESSION_REPORT_2026-03-27_DETECTION_DISCOVERY.md` | This report |
| `models/brightfield_cells_nuclei_v3_aligned/model_weights.pth` | InstanSeg v3.0 checkpoint |
