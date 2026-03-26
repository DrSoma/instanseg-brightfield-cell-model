# Session Report: Gen-2 Training Complete + Gap Validation

**Date**: 2026-03-26
**Status**: Gen-2 Megazord COMPLETE. Gen-3 multi-vector training starting.

---

## Gen-2 Final Results

### Training Summary

| Metric | Value |
|--------|-------|
| Architecture | WiderAttnUNet (48/96/192/384 + attention gates) |
| Parameters | 17.66M |
| Best epoch | 72 (of 92 total) |
| Best val_loss | 0.3484 |
| Peak membrane accuracy | 91% (epoch 19) |
| Stable membrane accuracy | 89-90% (epochs 60-92) |
| Stable cytoplasm accuracy | 88-89% |
| LR drops | 5 times (5e-4 → 2.5e-4 → 1.3e-4 → 6.3e-5 → 3.1e-5 → 1.6e-5) |
| Early stopping | Epoch 92, patience=20 exhausted (best was epoch 72) |
| Training time | ~64 hours (92 epochs × 42 min on NVMe) |
| Would have been on HDD | ~169 hours (92 × 110 min) — saved 105 hours |

### Gap Validation Results

| Method | Membrane DAB | Cytoplasm DAB | Gap | Cells |
|--------|-------------|---------------|-----|-------|
| **Gen-2 Argmax** | 0.4283 | 0.2191 | **+0.2092** | 4,307 |
| **Gen-2 Soft** | 0.5667 | 0.2900 | **+0.2767** | 5,591 |
| Gen-1 Argmax | 0.4265 | 0.2184 | +0.2081 | 4,233 |
| Gen-1 Soft | 0.6057 | 0.3131 | +0.2926 | 5,935 |
| Bar-filter | 0.2938 | 0.1669 | +0.1269 | 6,626 |
| Fixed ring | 0.2537 | 0.2738 | -0.0201 | 6,634 |

### Key Finding: Accuracy Improved, Measurement Plateaued

Gen-2 is a dramatically better pixel classifier than Gen-1:
- Membrane: 90% vs 71% (+19 points)
- Cytoplasm: 89% vs 65% (+24 points)
- Val loss: 0.3484 vs 0.5505 (36% lower)

But the clinical DAB gap barely moved:
- Argmax: +0.2092 vs +0.2081 (+0.001 improvement)
- Soft: +0.2767 vs +0.2926 (-0.016, slightly lower)

**Why:** The gap metric saturated at Gen-1. The extra 20% of membrane pixels that Gen-2 correctly classifies are borderline pixels with DAB values close to the cytoplasm average. Adding them doesn't change the mean DAB significantly.

**Implication:** The measurement quality is as good as it's going to get from pixel classification alone. Further improvement requires better thresholds (pathologist calibration) or better data (dilution slides, multi-scanner), not better models.

### Stage 3: Membrane-Aligned Masks

78,511 membrane-aligned cell masks generated using Gen-2's predictions:
- Used Gen-2's membrane probability map as energy landscape for watershed
- Boundaries snap to predicted membrane location instead of arbitrary midpoint
- Stored at: `data/masks_membrane_aligned/`
- Ready for InstanSeg v3.0 retraining

---

## Megazord Pipeline Complete Summary

| Stage | What | Time | Output |
|-------|------|------|--------|
| Stage 1 | Gen-1 → soft labels for 78K tiles | 3 sec (cached) | 78,511 .npy files |
| Stage 2 | Train Gen-2 WiderAttnUNet | 64 hours (92 epochs) | Best at epoch 72 |
| Gap validation | Argmax +0.209, Soft +0.277 | 18 sec | PASS |
| Stage 3 | Generate aligned masks | 50 min | 78,511 refined masks |
| **Total** | | **65 hours** | |

---

## Generation Comparison

| Model | Params | mem% | cyto% | Argmax Gap | Soft Gap | Training |
|-------|--------|------|-------|------------|----------|----------|
| Bar-filter | N/A | N/A | N/A | +0.127 | N/A | N/A |
| Gen-1 | 7.76M | 71% | 65% | +0.208 | +0.293 | 10 hrs |
| Gen-2 | 17.66M | 90% | 89% | +0.209 | +0.277 | 64 hrs |

**Conclusion:** Gen-1 is the breakthrough (classical → neural, 1.6x better than bar-filter). Gen-2 refined pixel classification substantially but didn't improve the measurement. Gen-2's value is in better heatmaps and the membrane-aligned masks for InstanSeg retraining.

---

## Infrastructure Achievements

| Optimization | Before | After | Improvement |
|-------------|--------|-------|-------------|
| Training storage | HDD (110 min/epoch) | NVMe (42 min/epoch) | 2.6x faster |
| Total time saved | — | ~105 hours | — |
| Swap management | Manual | Cron every 15 min | Automated |
| Checkpoint resume | Model weights only | Full state (optimizer + scheduler + patience) | Clean resume |
| LVM extension | 100 GB root | 500 GB root | 400 GB more NVMe |

---

## What's Next

### Immediate: Gen-3 Multi-Vector Training
- Goal: scanner/staining robustness (not better accuracy)
- Method: generate labels with multiple stain vector sets
- Train on combined labels so model handles staining variation

### After Gen-3:
1. Dilution slide validation (analytical sensitivity + linearity)
2. 100-cell spot check (physician annotation)
3. Pathologist calibration (Dr. Fiset scores 30-tile export)
4. Coloring-book heatmaps with Gen-2 for all slides
5. Re-run 191-slide cohort with new measurements
6. McGill tech transfer consultation
7. Test on Leica Aperio AT2 scans
