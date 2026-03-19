# Debate Gate 3: Model Selectivity Validation
## Date: 2026-03-19 | 5 Claude Agents | Codex Failed (401)

---

## Verdict: ALL 5 AGENTS — NEEDS MORE

---

## Agent Summaries

### Roche Diagnostics VP (Computational Pathology)
- 8 tests are good for research characterization, insufficient for CDx analytical validation
- **Missing: iCAP control concordance** — run on known-positive gastric mucosa vs known-negative stroma on same slide. Selectivity ratio (positive vs negative region) is the key metric.
- **Missing: reproducibility** — run same tiles 3x, CV < 15% per CLSI EP05
- **Missing: pathologist review of overlays** — no human in the loop
- 40% on blue tissue is manageable IF denominator handled correctly
- Two-model approach recommended: general detector for denominator, DAB model for numerator
- Membrane DAB scale mismatch is more urgent than selectivity question

### Paige AI CTO (Virchow2 Creator)
- Virchow2 IS the right tool but must use H-channel after deconvolution, NOT raw IHC
- KMeans K=2 is oversimplified — use K=4-5 for lymphocytes, fibroblasts, endothelial
- DAB validation of clusters is CIRCULAR (testing DAB selectivity using DAB as ground truth)
- Validate clusters with morphometrics (nuclear area, eccentricity) instead
- Virchow2 attention maps (layers 20-24) naturally segment tissue compartments
- Three-step protocol: morphometric triage → serial section test → sub-threshold expression check
- Serial section H&E is the single most definitive test

### Astellas Pharma Biomarker Strategy Head
- BC_ClassII 46%→98% is the most consequential number — drug access problem
- BUT: 277 vs 281 H-score means model can't distinguish moderate from strong = broken grading
- 40% detection on negative tissue: NOT acceptable for CDx (must be <5%)
- iCAP control showing 83.5% positive (vs 23% baseline) = threshold problem extends to controls
- Three non-negotiable gates: threshold recalibration, pathologist concordance, specificity demonstration
- VENTANA protocol requires "tumor cells" in denominator — model doesn't distinguish tumor vs non-tumor

### Edinburgh InstanSeg Author (Dr. Goldsborough)
- Blue-tissue detection is EXPECTED from architecture:
  1. Shared encoder pretrained on all nuclei — still fires on cell-like morphology
  2. Cells decoder initialized from nuclei head — retains nucleus detection inertia
  3. Loss never penalized false positives on blue tissue (no negative examples in training)
- Seed logits at 1.6 (vs stock 6.9): model not converged, noisy GT from permissive filter
- **Blue detections are primarily from NUCLEUS head** — check both channels separately
  - Predicts: nuclei channel = 80-90% of baseline on blue, cells channel = 30-40%
- Heatmaps: seed map overlaid on DAB channel (not RGB) is the most informative visualization
  - If seeds correlate with DAB → partial stain specificity
  - If seeds correlate with nuclear density regardless of DAB → no specificity
- v2.0 with stricter filtering + 78k tiles should push seed logits to 4.0+

### MUHC Pathologist (Dr. Fiset)
- 97.7% on ClassII does NOT match visual assessment — expects 60-70%
- 0% negative cells is biologically impossible on a moderate slide
- Blue-tissue detections likely mix of: real unstained epithelial, tissue folds, mucin edges, gland lumens
- Needs overlays with intensity-coded boundaries (neg/1+/2+/3+ colors)
- **Red flag**: if iCAP control and patient tissue return same H-score = system measuring wrong thing
- "The 97.7% is an artifact of uncalibrated DAB measurement, not a biological finding"

---

## Consensus Actions

### Add 3 Tests from Debate:
9. **Separate nuclei vs cells channel analysis** — Edinburgh recommendation
10. **iCAP paired concordance** — Roche recommendation (same slide, positive vs negative region)
11. **Virchow2 with H-channel preprocessing** — Paige recommendation

### Fix Existing Tests:
- Test 5: Use K=4-5 not K=2; validate with morphometrics not DAB; preprocess with H-channel
- Test 8: Overlay seeds on DAB channel (not RGB) as additional panel
- All visual tests: color-code boundaries by intensity grade

### Blockers Before Clinical Use:
1. Threshold recalibration (membrane-ring DAB → pathologist grade mapping)
2. Pathologist 30-tile concordance (non-negotiable)
3. Reproducibility assessment (CV < 15%)
4. Denominator fix (two-model approach or negative-example retraining)
