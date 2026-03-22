# Concerns Documentation Guide

**Source**: 15-agent project review (9 investigative + 5 debate + 1 quality gate)
**Date**: 2026-03-21
**Purpose**: Every concern raised, with exact file locations, clear explanations, and recommended fixes

---

## Table of Contents

1. [CONCERN-01: Uncalibrated Scoring Thresholds](#concern-01)
2. [CONCERN-02: Normal Gastric Mucosa Inflates 3+ Scores](#concern-02)
3. [CONCERN-03: DL Model Does Not Trace Membranes — Bar-Filter Does](#concern-03)
4. [CONCERN-04: Negative Correlation with Baseline (r = -0.32)](#concern-04)
5. [CONCERN-05: REB Says "Tumour Cells" but Pipeline Scores All Cells](#concern-05)
6. [CONCERN-06: Single-Site, Single-Reader, Single-Scanner Validation](#concern-06)
7. [CONCERN-07: McGill Tech Transfer / IP Not Protected](#concern-07)
8. [CONCERN-08: Stain Augmentation Ablation Is Invalid](#concern-08)
9. [CONCERN-09: Training Has Zero Segmentation Quality Feedback](#concern-09)
10. [CONCERN-10: Cell F1 = 0.24 Against Self-Supervised Labels](#concern-10)
11. [CONCERN-11: H&E Cross-Reactivity in Stain Deconvolution](#concern-11)
12. [CONCERN-12: 20% Training Success Rate](#concern-12)
13. [CONCERN-13: Hardcoded Paths (40+ instances)](#concern-13)
14. [CONCERN-14: Four Near-Identical Training Scripts](#concern-14)
15. [CONCERN-15: Clinical Thresholds Scattered Across Files](#concern-15)
16. [CONCERN-16: compute_hscore() Duplicated 6 Times](#concern-16)
17. [CONCERN-17: No Requirements Lockfile](#concern-17)
18. [CONCERN-18: Test Suite Cannot Run](#concern-18)
19. [CONCERN-19: No Pipeline Orchestration](#concern-19)
20. [CONCERN-20: No Experiment Tracking](#concern-20)
21. [CONCERN-21: Dataset Subsampling Overwrites the .pth File](#concern-21)
22. [CONCERN-22: Exception Swallowing in Mask Generation](#concern-22)
23. [CONCERN-23: O(n*m) IoU Computation in Evaluation](#concern-23)
24. [CONCERN-24: No Artifact Detection](#concern-24)
25. [CONCERN-25: Deterministic Mode Slows Inference](#concern-25)
26. [CONCERN-26: DataLoader num_workers = 0 Starves GPU](#concern-26)
27. [CONCERN-27: No torch.compile for Inference](#concern-27)
28. [CONCERN-28: No Schema Contracts Between Pipeline Stages](#concern-28)
29. [CONCERN-29: No End-to-End Data Lineage](#concern-29)
30. [CONCERN-30: UNCALIBRATED Disclaimer Is Structurally Inadequate](#concern-30)
31. [CONCERN-31: No Docker / Containerization](#concern-31)
32. [CONCERN-32: No CI/CD Pipeline](#concern-32)
33. [CONCERN-33: Monkey-Patching InstanSeg Internals](#concern-33)
34. [CONCERN-34: No Spatial Heterogeneity Analysis](#concern-34)
35. [CONCERN-35: Cell Count Discrepancy Between Baseline and Model](#concern-35)

---

## CONCERN-01: Uncalibrated Scoring Thresholds {#concern-01}

**Severity**: BLOCKING
**Flagged by**: Agents 2, 3, 5, 6, 9 + Debate D2, D3

### What is the problem?

The composite grade thresholds that determine whether a cell is scored 0, 1+, 2+, or 3+ are currently set to fixed DAB optical density values that have not been validated against pathologist scoring.

### Where exactly?

The thresholds `(0.10, 0.20, 0.35)` are defined in four files:

| File | Line |
|------|------|
| `scripts/11_add_membrane_columns.py` | 69 |
| `scripts/11b_add_membrane_fast.py` | 48 |
| `scripts/11c_membrane_pipeline_dualgpu.py` | 50 |
| `scripts/12_eligibility_analysis.py` | 63 |

The eligibility threshold (>=75% of cells at >=2+) is in:

| File | Line |
|------|------|
| `scripts/12_eligibility_analysis.py` | 58-59 |

### What do the thresholds mean?

```
Grade 0 (negative):  membrane DAB OD < 0.10
Grade 1+ (weak):     0.10 <= DAB OD < 0.20
Grade 2+ (moderate): 0.20 <= DAB OD < 0.35
Grade 3+ (strong):   DAB OD >= 0.35
```

These numbers were chosen as starting estimates, not derived from pathologist ground truth.

### Why does it matter?

On iCAP control slides (BC_ClassII, BC_ClassIII), 82% of cells score as 3+. This means nearly every cell exceeds the 0.35 threshold. Because the eligibility criterion is >=75% at >=2+, almost any slide with DAB-positive cells becomes "eligible." This produces a 30.4% eligibility rate compared to the baseline's 0.9%.

The "16 patients gained eligibility" finding depends entirely on these thresholds. Different thresholds could produce 0 gained or 50 gained.

### Evidence

From `evaluation/eligibility_analysis/eligibility_summary.json`:
```json
"thresholds": {
    "grade_thresholds_dab_od": [0.1, 0.2, 0.35],
    "eligibility_pct": 75.0,
    "eligibility_grade": ">=2+",
    "calibrated": false
}
```

The file includes the disclaimer:
```
"THRESHOLDS_UNCALIBRATED -- RESEARCH USE ONLY. Do NOT use for clinical decisions."
```

### How to fix

1. Have Dr. Fiset score the 30-tile pathologist export (already prepared in `evaluation/pathologist_export/`)
2. Map her 0/1+/2+/3+ grades to the DAB OD values measured by the pipeline on the same cells
3. Fit threshold boundaries that maximize concordance with pathologist grades
4. Update all four files above with the calibrated values
5. Re-run the eligibility analysis with calibrated thresholds

---

## CONCERN-02: Normal Gastric Mucosa Inflates 3+ Scores {#concern-02}

**Severity**: HIGH
**Flagged by**: Post-review analysis (user + agents combined)

### What is the problem?

Normal gastric epithelium physiologically expresses CLDN18.2 at high levels. On CLDN18.2 IHC, normal gastric mucosa shows strong, circumferential membrane staining — a textbook 3+ pattern. The pipeline scores ALL epithelial cells with DAB stain, including normal mucosa. This means:

- On iCAP controls (which contain normal gastric mucosa), the 82% at 3+ may be **biologically correct**, not a calibration error
- On patient tissue, normal gastric mucosa adjacent to tumor would contribute 3+ scores, inflating the overall positivity rate
- The "16 patients gained eligibility" could be partially driven by normal mucosa inclusion

### Evidence

From the REB (3rd draft):
```
"normal gastric mucosa (physiological CLDN18.2 expression)"
```

This is explicitly used as the POSITIVE CONTROL because it IS strongly positive.

From `memory/reference_clinical_context.md`:
```
Normal gastric mucosa = CLDN18.2 positive (physiological expression) → our POSITIVE control
```

### Why this matters more than the agents realized

The agents flagged "no tumor cell classification" based on VENTANA CDx criteria. The real reason it matters is biological: **normal gastric epithelium is naturally 3+ for CLDN18.2, and including it inflates every metric.** This is not about regulatory compliance — it's about measurement validity.

### How to investigate

1. Take 3-5 of the "16 gained eligibility" slides
2. Overlay the per-cell heatmaps on the tissue
3. Check whether the high-scoring (3+) regions correspond to normal gastric mucosa or actual tumor
4. If normal mucosa is driving the scores, that confirms this concern

### How to fix

Options (in order of increasing effort):
1. Use the existing region annotations to restrict scoring to `Patient_Tissue` regions only (exclude iCAP controls) — this is already done in `12_eligibility_analysis.py`
2. Within `Patient_Tissue`, use a simple morphology-based classifier (gland architecture vs solid tumor) to separate normal mucosa from adenocarcinoma
3. Train a tissue-type classifier (normal mucosa vs tumor vs stroma) — more robust but requires annotated data

---

## CONCERN-03: DL Model Does Not Trace Membranes — Bar-Filter Does {#concern-03}

**Severity**: HIGH (framing issue)
**Flagged by**: Agents 6, 9 + Debate D2

### What is the problem?

The InstanSeg model produces cell boundary polygons. When you measure DAB intensity in a simple ring around those boundaries ("fixed 10px ring"), the membrane DAB is actually LOWER than the cytoplasm DAB. The model's boundaries do not preferentially follow the membrane staining.

Only when the Aperio-style bar-filter post-processor is applied does the membrane signal emerge.

### Evidence

From `evaluation/boundary_validation.json`:

**Fixed 10px ring — FAILS:**
```json
"fixed_10px_ring": {
    "membrane_dab_mean": 0.2537,
    "cytoplasm_dab_mean": 0.2738,
    "gap": -0.0201,
    "status": "FAIL"
}
```

The membrane ring has LOWER DAB than the cytoplasm. The model's cell boundaries are not membrane-aligned.

**Bar-filter — PASSES:**
```json
"aperio_bar_filter": {
    "membrane_dab_mean": 0.2938,
    "cytoplasm_dab_mean": 0.1669,
    "gap": 0.1269,
    "status": "PASS"
}
```

The bar-filter correctly identifies membrane signal (gap = +0.127 OD).

### What this means

The CLDN18.2 scoring depends on the bar-filter (classical image processing), not on the deep learning model's boundaries. The DL model provides cell masks for extracting per-cell regions, but the actual membrane measurement is done by the bar-filter weighting.

### Why this matters for publications

If you write "deep learning membrane segmentation," peer reviewers will expect the DL model to find membranes. It doesn't. The accurate framing is "DL-assisted cell segmentation combined with bar-filter membrane quantification."

### How to fix

1. **Framing**: Describe the system as a two-stage pipeline (DL cell detection + classical membrane measurement)
2. **Future improvement**: Train a membrane-aware loss function that penalizes boundary predictions that don't align with DAB gradients

---

## CONCERN-04: Negative Correlation with Baseline (r = -0.32) {#concern-04}

**Severity**: HIGH (uninvestigated)
**Flagged by**: Agents 2, 3, 9 + Debate D2, D3, D5

### What is the problem?

When comparing the pipeline's %2+ scores against the baseline pipeline's %2+ scores on 95 matched slides, the Pearson correlation is **negative** (r = -0.32). A negative correlation means: when the baseline says a slide has high positivity, the new pipeline says it has low positivity, and vice versa.

### Evidence

From `evaluation/eligibility_analysis/eligibility_summary.json`:
```json
"comparison": {
    "n_matched_slides": 95,
    "pct_2plus_correlation": -0.3207,
    "h_score_correlation": -0.2374,
    "h_score_mae": 73.77
}
```

### Why it might be expected (not a problem)

The baseline measures DAB intensity over the **nucleus** (because it only has nuclei polygons with a fixed expansion). The new pipeline measures DAB over the **membrane ring** (because it has cell polygons + bar-filter).

For a membrane marker like CLDN18.2:
- A cell with strong membrane staining but light nucleus → baseline says "low," new model says "high"
- A cell with nuclear DAB bleed-through but weak membrane → baseline says "high," new model says "low"

Measuring different cell compartments can naturally produce anti-correlated results. This would **validate** that the model is measuring something different (membrane vs nucleus).

### Why it might be a problem

If both methods are supposed to measure CLDN18.2 expression (even in different compartments), you'd expect the biological signal to correlate — high CLDN18.2 expression means both membrane and nuclear regions should have more DAB (due to optical bleed). A negative correlation could indicate a systematic measurement error in one or both systems.

### How to investigate

1. Pull 5-10 of the most anti-correlated slides (high baseline, low model or vice versa)
2. Examine the tissue visually: is the baseline measuring nuclear bleed-through? Is the model's bar-filter missing weak membrane signal?
3. Check whether the negative correlation persists when using matched thresholds (same DAB OD cutoffs for both pipelines)
4. Check whether cell count differences (baseline mean 820k vs model mean 476k) drive the correlation via Simpson's paradox

### How to fix

This is an investigation, not a code fix. Once the root cause is understood, either:
- Document it as expected behavior (different compartments) with a clear explanation
- Fix the underlying measurement if it reveals a bug

---

## CONCERN-05: REB Says "Tumour Cells" but Pipeline Scores All Cells {#concern-05}

**Severity**: MODERATE
**Flagged by**: Agents 3, 9 + post-review analysis

### What is the problem?

The REB (3rd draft) states:
```
"InstanSeg will be applied to segment tumour cells on digitized whole‑section slides"
```

The pipeline segments all cells with DAB staining, not specifically tumour cells. This includes normal gastric mucosa (which is CLDN18.2-positive physiologically), stroma, and inflammatory cells.

### Nuance

The REB positions the tool as QA, not diagnostic. The inclusion criteria is "All cases" with no exclusion. The iCAP control design explicitly uses normal gastric mucosa as the positive control. So the pipeline's behavior is consistent with the QA purpose — but the REB language specifically says "tumour cells."

### How to address

Options:
1. **Amend the REB language** to "segment cells" (since it's QA, this is defensible)
2. **Add a tissue classifier** to distinguish tumour from normal epithelium
3. **Document the discrepancy** as a known limitation with justification (QA use case doesn't require tumor-only segmentation)

Discuss with Dr. Fiset which approach fits the REB committee's expectations.

---

## CONCERN-06: Single-Site, Single-Reader, Single-Scanner Validation {#concern-06}

**Severity**: BLOCKING (for publication or deployment)
**Flagged by**: Agents 2, 3, 5, 9

### What is the problem?

All 191 slides are from MUHC, scanned on Hamamatsu NanoZoomer, stained with the same MUHC protocol, and reviewed by one pathologist (Dr. Fiset). There is zero evidence the pipeline works on:
- Slides from another institution
- Slides scanned on a different scanner (e.g., Leica Aperio AT2 — mentioned in the REB)
- Slides stained with a different protocol
- Slides scored by a different pathologist

### Why it matters

The stain deconvolution uses fixed vectors calibrated for this specific scanner/stain combination (`config/default.yaml` lines 20-28). A different scanner's color profile would shift all DAB measurements.

The REB explicitly mentions both Hamamatsu NanoZoomer S360 and Leica Aperio AT2, suggesting multi-scanner validation is expected.

### How to fix

1. **Short-term**: Scan 10-20 slides on the Leica Aperio AT2 (already available per REB) and compare measurements
2. **Medium-term**: Recruit 2 additional GI pathologists to score 30+ cases independently for inter-reader kappa
3. **Long-term**: Obtain 20-30 slides from a second institution

---

## CONCERN-07: McGill Tech Transfer / IP Not Protected {#concern-07}

**Severity**: URGENT (time-sensitive)
**Flagged by**: Agents 3, 5 + Debate D4

### What is the problem?

The zero-annotation self-supervised approach to membrane IHC scoring is potentially novel and patentable. No invention disclosure has been filed with McGill. Any public disclosure (publication, preprint, conference poster, even a detailed talk) before filing would compromise the novelty window.

### Evidence

From the REB:
```
"Any intellectual property arising from this work will be held or divided
according to MUHC institutional policies and existing research agreements"
```

The Astellas funding agreement may also have IP clauses.

### How to fix

1. Contact McGill's Research Agreements Office
2. File an invention disclosure describing the self-supervised training method
3. Do NOT submit any preprint, abstract, or poster until IP counsel advises
4. Check the Astellas funding agreement for IP obligations

---

## CONCERN-08: Stain Augmentation Ablation Is Invalid {#concern-08}

**Severity**: MODERATE
**Flagged by**: Agents 2, 6 + Debate D2

### What is the problem?

The project ran four training variants to test stain augmentation:
- v2.0a: half-strength stain aug
- v2.0b: no stain aug
- v2.0c: strong stain aug → 2.6 cells/tile
- v2.0d: zero stain aug → 0 cells/tile

The "finding" was that stain augmentation is harmful (zero > strong > weak).

**All four models suffered sigma collapse** because they were missing the `--pretrained-folder` flag. The sigma collapse caused all models to produce 0 to 2.6 cells per tile, compared to v1.0's 150-248. The stain augmentation variable is completely confounded by the missing pretrained weights.

### Evidence

From `evaluation/stain_aug_ablation.json`, all v2.0 variants show near-zero cell counts. The comparison is between broken models.

From `evaluation/DEBATE_GATE_5_V2_BUG_FIX.md`, the root cause of sigma collapse was identified as missing pretrained weight initialization, not stain augmentation.

### What to do

1. Do NOT cite this ablation as evidence in any publication
2. If stain augmentation effects are important, re-run the ablation with v2.1 (which has the pretrained weights + sigma clamp fix), varying only stain augmentation
3. Update `memory/finding_stain_aug_ablation.md` to note the confound

---

## CONCERN-09: Training Has Zero Segmentation Quality Feedback {#concern-09}

**Severity**: HIGH
**Flagged by**: Agents 2, 6

### What is the problem?

The `_safe_ap` monkey-patch in every training script catches ALL metric computation errors and returns `(0.0, 0.0)`. This means:
- The training loop never sees real F1 scores
- Early stopping is based on loss only, not segmentation quality
- A model can train for 200 epochs producing zero cells while loss decreases (this is exactly what happened with v2.0)

### Where exactly?

**scripts/04_train.py** — Lines 242-256:
```python
def _safe_ap(*a, **kw):
    try:
        return _orig_ap(*a, **kw)
    except (IndexError, RuntimeError, ValueError):
        return (0.0, 0.0)
```

Same pattern in:
- `scripts/04_train_v2.py` — Lines 606-612
- `scripts/04_train_v2b_no_stain_aug.py` — Lines 608-614
- `scripts/04_train_v2c_strong_stain_aug.py` — Lines 609-615
- `scripts/04_train_v2d_zero_stain_aug.py` — Similar location

### Why the patch exists

InstanSeg's `_robust_average_precision` crashes on dual-head (nuclei + cells) outputs due to a shape mismatch in `torch_sparse_onehot`. Without the patch, training crashes. With it, training continues but flies blind.

### How to fix

1. Fix the upstream `_robust_average_precision` for dual-head mode (report bug to InstanSeg)
2. Add a periodic inference check: every N epochs, run the model on 10 validation tiles and count cells. If cell count = 0, halt training.
3. Log sigma statistics every epoch to detect collapse early

---

## CONCERN-10: Cell F1 = 0.24 Against Self-Supervised Labels {#concern-10}

**Severity**: HIGH
**Flagged by**: Agents 2, 6

### What is the problem?

When the trained model's predictions are compared against the watershed-generated training labels, the cell-level F1 score is only 0.24. The model disagrees with its own training signal 76% of the time for cells.

### Evidence

From `evaluation/metrics.json`:
```json
"cells": {
    "f1_mean": 0.157,
    "f1_micro": 0.244,
    "precision": 0.212,
    "recall": 0.288,
    "dice": 0.691,
    "tp": 58763,
    "fp": 218322,
    "fn": 145466
}
```

For comparison, nuclei are better:
```json
"nuclei": {
    "f1_mean": 0.338,
    "f1_micro": 0.495,
    "precision": 0.407,
    "recall": 0.632,
    "dice": 0.848
}
```

Evaluation was run on 1,781 tiles with 0 errors.

### Why it might not be as bad as it looks

- Watershed pseudo-labels for cells are noisy (over-segment touching cells, jagged boundaries)
- The Dice score is 0.691 for cells — meaning pixel-level overlap is decent even if instance matching is poor
- A low F1 against noisy labels doesn't necessarily mean the model is wrong — it might be producing smoother, more biologically realistic boundaries
- The pathologist validated downstream output as accurate (97-98% positivity)

### Why it might actually be concerning

- 218,322 false positives vs 58,763 true positives means the model OVER-segments (3.7x more FP than TP)
- The cell head may be producing fragmented, noisy boundaries rather than coherent cell shapes
- This connects to CONCERN-03: the cell boundaries don't trace membranes, which is consistent with the cell head not learning well

### How to investigate

1. Visualize 20-30 tiles with predicted cell masks overlaid on the image
2. Compare against watershed labels side-by-side
3. Determine if the model is over-segmenting (splitting cells) or producing biologically wrong shapes

---

## CONCERN-11: H&E Cross-Reactivity in Stain Deconvolution {#concern-11}

**Severity**: MODERATE
**Flagged by**: Agents 6, 9 + Debate D2

### What is the problem?

When the pipeline's stain deconvolution (calibrated for hematoxylin + DAB) is applied to an H&E slide (which contains eosin, not DAB), it reports a mean "DAB" optical density of 0.363. This happens because eosin's color profile partially overlaps with DAB in RGB space, and the deconvolution math cannot distinguish them.

### Evidence

From `evaluation/comprehensive_validation/he_negative_control.json`:
```json
{
    "slide": "/media/fernandosoto/DATA/Cluster 1/BTC-01.ndpi",
    "stain": "H&E (no DAB)",
    "tiles": 50,
    "mean_dab_on_he": 0.3632,
    "bar_filter_gap_he": 0.1396,
    "bar_filter_gap_he_std": 0.052
}
```

### What this means practically

On your actual CLDN18.2 IHC slides, this is less alarming than it sounds:
- IHC slides use **hematoxylin counterstain**, not eosin. The background "noise" from hematoxylin bleeding into the DAB channel is much lower than eosin.
- The 0.363 OD is on H&E tissue with intense eosinophilic cytoplasm — a worst case.
- The bar-filter gap (0.140) on H&E is still positive, meaning the bar-filter itself can produce false "membrane" signal from eosinophilic patterns.

However, it establishes that the DAB measurement has a non-zero noise floor. Any cell with DAB OD close to 0.10 (the Grade 1+ threshold) might be noise rather than real signal.

### How to investigate

Run the deconvolution on **stroma regions** of your IHC slides (where there should be minimal CLDN18.2 DAB). The mean DAB OD in stroma gives you the actual noise floor for your tissue type.

### How to fix

1. Measure the noise floor on IHC stroma and document it
2. Consider setting the Grade 0/1+ threshold above the measured noise floor
3. For maximum robustness, consider per-slide adaptive stain normalization (Macenko or Vahadane method) instead of fixed vectors

---

## CONCERN-12: 20% Training Success Rate {#concern-12}

**Severity**: MODERATE
**Flagged by**: Agents 6 + Debate D2

### What is the problem?

Training v2.0/v2.1 required 5 attempts to produce a working model:
1. v2.0a (half stain aug) — sigma collapse, 0 cells
2. v2.0b (no stain aug) — sigma collapse, 0 cells
3. v2.0c (strong stain aug) — sigma collapse, 2.6 cells/tile
4. v2.0d (zero stain aug) — sigma collapse, 0 cells
5. v2.1 (gradual sigma clamp + pretrained weights + no hotstart) — **success**

The successful recipe requires a specific combination:
- v1.0 fine-tuned weights as initialization
- Gradual sigma clamp schedule: -2.0 (epochs 0-19) → -2.5 (20-39) → -3.0 (40-69) → -3.5 (70-99) → -4.0 (100+)
- No hotstart
- Zero stain augmentation

### Where is the fix implemented?

`scripts/04_train_v2d_zero_stain_aug.py` — Lines 592-636:
```python
SIGMA_SCHEDULE = [
    (0,   -2.0),   # epochs 0-19: tight clamp
    (20,  -2.5),   # epochs 20-39: slightly relaxed
    (40,  -3.0),   # epochs 40-69: moderate
    (70,  -3.5),   # epochs 70-99: loose
    (100, -4.0),   # epochs 100+: very loose (safety net only)
]
```

### Why it matters

This schedule was discovered empirically. It has not been demonstrated to work reproducibly across different random seeds or datasets. For a clinical tool, a training pipeline that fails 80% of the time is a regulatory concern (ISO 13485 design controls).

### How to fix

1. Re-run training with 3 different random seeds to confirm the schedule is robust
2. Add the sigma monitoring recommended by the debate consensus (log sigma mean/std every epoch)
3. Add an automated health check: if F1 = 0 after epoch 10, halt and alert

---

## CONCERN-13: Hardcoded Paths (40+ instances) {#concern-13}

**Severity**: HIGH (portability)
**Flagged by**: Agents 1, 4, 8

### What is the problem?

At least 17 scripts contain hardcoded absolute paths to this specific machine. Nobody else can run the pipeline without find-and-replace across multiple files.

### Where exactly?

Key hardcoded paths:

| Path | Files |
|------|-------|
| `/home/fernandosoto/Documents/models/...` | `07_validate_boundaries.py:36`, `filter_branch_j.py:115`, `resolve_debate_issues.py:158` |
| `/home/fernandosoto/claudin18_venv/bin/python` | `11_add_membrane_columns.py:1` (shebang), `12_eligibility_analysis.py:1` (shebang) |
| `/media/fernandosoto/DATA/CLDN18 slides` | `11b`, `11c`, `13`, `14` |
| `/pathodata/Claudin18_project/...` | `10_icap_concordance.py:35-36`, `12_eligibility_analysis.py:67` |
| `/tmp/cohort_v1/cell_data` | `11_add_membrane_columns.py:635`, `11b:394`, `12:810`, `13:200` |
| `/tmp/pipeline_comparison/...` | `07_pathologist_export.py:51`, `08:254`, `09:293`, `10:330` |

### How to fix

1. Add a `[paths]` section to `config/default.yaml` with environment variable defaults:
   ```yaml
   paths:
       cohort_dir: ${COHORT_DIR:=data/cohort_v1/cell_data}
       comparison_dir: ${COMPARISON_DIR:=data/pipeline_comparison}
       model_weights: ${MODEL_WEIGHTS:=models/brightfield_cells_nuclei_v1_finetuned/model_weights.pth}
       baseline_csv: ${BASELINE_CSV:=/pathodata/Claudin18_project/exports/cldn18_region_summary.csv}
   ```
2. Replace all hardcoded paths with config lookups
3. Replace shebang lines with `#!/usr/bin/env python3`
4. Create the promised `.env.example` file

---

## CONCERN-14: Four Near-Identical Training Scripts {#concern-14}

**Severity**: MODERATE (technical debt)
**Flagged by**: Agents 1, 4

### What is the problem?

Four training scripts share ~95% of their code:
- `scripts/04_train_v2.py` (819 lines)
- `scripts/04_train_v2b_no_stain_aug.py` (821 lines)
- `scripts/04_train_v2c_strong_stain_aug.py` (822 lines)
- `scripts/04_train_v2d_zero_stain_aug.py` (906 lines)

The only differences are stain augmentation settings and (in v2d) the sigma clamp. A bug fix in any shared logic would need to be applied to all four files.

### How to fix

Consolidate into one script with CLI flags:
```bash
python scripts/04_train_v2.py --stain-aug none     # v2b
python scripts/04_train_v2.py --stain-aug strong    # v2c
python scripts/04_train_v2.py --stain-aug zero --sigma-clamp  # v2d/v2.1
```

---

## CONCERN-15: Clinical Thresholds Scattered Across Files {#concern-15}

**Severity**: HIGH (governance risk)
**Flagged by**: Agents 1, 4, 8

### What is the problem?

The grade thresholds `(0.10, 0.20, 0.35)` are defined as module-level constants in four separate files (see CONCERN-01). The eligibility criteria (75%, >=2+) are in another file. The DAB positivity threshold (0.05 for membrane coverage) is in `quality.py:113`. The membrane ring parameters (`RING_ERODE_K=5`) are in `11_add_membrane_columns.py`.

If any threshold is updated during calibration, it must be changed in 4+ files simultaneously with no automated consistency check.

### How to fix

Create `config/clinical_thresholds.yaml`:
```yaml
grading:
    grade_1_min: 0.10  # UNCALIBRATED
    grade_2_min: 0.20  # UNCALIBRATED
    grade_3_min: 0.35  # UNCALIBRATED
    calibrated: false
eligibility:
    pct_threshold: 75.0
    min_grade: 2
membrane:
    ring_erode_ksize: 5
    ring_erode_iterations: 2
    dab_positive_threshold: 0.05
```

All scripts read from this single file.

---

## CONCERN-16: compute_hscore() Duplicated 6 Times {#concern-16}

**Severity**: LOW (code hygiene)
**Flagged by**: Agent 4

### What is the problem?

The same H-score computation is copy-pasted in 6 files:

| File | Line |
|------|------|
| `scripts/12_eligibility_analysis.py` | 97 |
| `scripts/10_icap_concordance.py` | 39 |
| `scripts/08_morphometric_filter.py` | 37 |
| `scripts/measure_membrane_thickness.py` | 291 |
| `scripts/run_pipeline_comparison.py` | 200 |
| `scripts/comprehensive_membrane_validation.py` | 320 |

### How to fix

Move to `src/instanseg_brightfield/scoring.py` and import everywhere:
```python
from instanseg_brightfield.scoring import compute_hscore
```

---

## CONCERN-17: No Requirements Lockfile {#concern-17}

**Severity**: MODERATE (reproducibility)
**Flagged by**: Agents 1, 5

### What is the problem?

`pyproject.toml` specifies minimum versions (`torch>=2.0`, `numpy>=1.24`) but there is no pinned lockfile. Different minor versions of scipy, scikit-image, or torch can produce different watershed or inference results.

### How to fix

```bash
pip freeze > requirements-lock.txt
```

Or adopt `uv` for reproducible environments:
```bash
uv pip compile pyproject.toml -o requirements-lock.txt
```

---

## CONCERN-18: Test Suite Cannot Run {#concern-18}

**Severity**: MODERATE
**Flagged by**: Agent 1

### What is the problem?

The project has 622 lines of tests across 6 files in `tests/`, but pytest is not installed in the active environment. Running `python -m pytest` fails with "No module named pytest."

### What tests exist

| Test file | Coverage |
|-----------|----------|
| `tests/test_watershed.py` | Watershed segmentation (245 lines) |
| `tests/test_quality.py` | Cell quality filtering (96 lines) |
| `tests/test_stain.py` | Stain deconvolution (66 lines) |
| `tests/test_config.py` | Config loading/hashing (75 lines) |
| `tests/test_pipeline_state.py` | Manifest/resumability (52 lines) |
| `tests/conftest.py` | Shared fixtures (88 lines) |

### What has zero test coverage

All 46 scripts in `scripts/` (28,000 lines). No integration tests.

### How to fix

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## CONCERN-19: No Pipeline Orchestration {#concern-19}

**Severity**: MODERATE
**Flagged by**: Agents 1, 4

### What is the problem?

14 numbered scripts (00-14) must be run manually in sequence. There is no Makefile, Snakemake workflow, or master script. If step 02 silently fails for some slides, step 03 packages incomplete data, and the error cascades.

### How to fix (simplest)

Create a `Makefile`:
```makefile
all: tiles masks dataset train evaluate export

tiles: data/tiles/manifest.json
data/tiles/manifest.json:
	python scripts/01_extract_tiles.py

masks: data/masks/manifest.json
data/masks/manifest.json: data/tiles/manifest.json
	python scripts/02_generate_masks.py
# ... etc
```

---

## CONCERN-20: No Experiment Tracking {#concern-20}

**Severity**: MODERATE
**Flagged by**: Agents 1, 6, 8

### What is the problem?

`pyproject.toml` lists `wandb>=0.15` and `tensorboard>=2.14` as optional dependencies, but neither is imported or used anywhere in the training code. Training metrics go to stdout and are not persisted in any structured format.

### Why it matters

With 5+ training variants, there is no way to compare experiments after the fact. The stain augmentation ablation failure (CONCERN-08) is a direct consequence — experiment tracking would have immediately shown that all v2.0 variants produced zero cells.

### How to fix

Add W&B or TensorBoard logging to the training script. Minimum viable:
```python
import wandb
wandb.init(project="instanseg-brightfield", config=cfg)
wandb.log({"loss": loss, "f1": f1, "sigma_mean": sigma.mean()})
```

---

## CONCERN-21: Dataset Subsampling Overwrites the .pth File {#concern-21}

**Severity**: MODERATE
**Flagged by**: Agent 1

### What is the problem?

When the dataset exceeds 12,000 items, `04_train.py` (lines 167-192) subsamples to ~10,000 items and **overwrites the original `.pth` file on disk**. This is destructive — re-running training modifies the dataset artifact.

### Where exactly

`scripts/04_train.py` — Line 190:
```python
_t.save(_ds, dataset_path)  # Overwrites original!
```

### How to fix

Save the subsampled version to a separate file:
```python
_t.save(_ds, dataset_path.with_stem(dataset_path.stem + "_subsampled"))
```

Or subsample at the DataLoader level instead of modifying the file.

---

## CONCERN-22: Exception Swallowing in Mask Generation {#concern-22}

**Severity**: MODERATE
**Flagged by**: Agents 1, 4

### What is the problem?

In `02_generate_masks.py` (lines 481-487), if a CPU post-processing worker raises any exception, it is logged but the tile is silently skipped:

```python
def _finalise_future(fut: Future, tile_id: str) -> None:
    try:
        filt_nuc, filt_cell, _fstats, tile_stats = fut.result()
    except Exception:
        logger.exception("CPU postprocess failed for %s", tile_id)
        return  # Silently skips this tile
```

There is no retry, no failed-tiles manifest, and no downstream notification.

### How to fix

1. Record failed tiles in a `failed_tiles.json` manifest
2. Add a summary at the end of processing: "X tiles failed out of Y total"
3. Optionally add retry logic with exponential backoff

---

## CONCERN-23: O(n*m) IoU Computation in Evaluation {#concern-23}

**Severity**: LOW (performance only)
**Flagged by**: Agent 1

### What is the problem?

`scripts/05_evaluate.py` (lines 32-61) computes pairwise IoU with nested Python loops:

```python
for i, pid in enumerate(pred_ids):
    pred_mask = pred == pid
    for j, gid in enumerate(gt_ids):
        gt_mask = gt == gid
        intersection = np.logical_and(pred_mask, gt_mask).sum()
```

For tiles with hundreds of instances, this is O(n*m) with per-pixel comparisons.

### How to fix

Use a sparse overlap matrix approach:
```python
from scipy.sparse import csr_matrix
overlap = csr_matrix((np.ones(H*W), (pred.ravel(), gt.ravel())), shape=(n_pred+1, n_gt+1))
```

---

## CONCERN-24: No Artifact Detection {#concern-24}

**Severity**: MODERATE (clinical use)
**Flagged by**: Agent 9

### What is the problem?

The pipeline has no detection or exclusion for common slide artifacts:

| Artifact | Effect on DAB | Detection |
|----------|--------------|-----------|
| Folded tissue | Doubles DAB intensity (double thickness) → inflates 3+ scores | None |
| Pen marks | High-OD artifacts → false positive DAB | None |
| Necrotic tissue | Non-specific DAB binding → false positives | None |
| Crush artifacts | Altered morphology and staining | None |
| Mucin pools | Acellular areas | Partially avoided by cell detection |

### Where tissue detection currently lives

`src/instanseg_brightfield/tissue.py` — uses brightness-only thresholding (threshold=235) on HSV V channel. This correctly separates tissue from background but has no artifact filtering.

### How to fix (progressive)

1. **Quick**: Add a fold detector based on local OD variance (folds have very high OD + high variance)
2. **Medium**: Add a pen mark detector using color thresholding in HSV (pen marks have distinctive hues)
3. **Thorough**: Train a simple CNN artifact classifier on annotated tiles

---

## CONCERN-25: Deterministic Mode Slows Inference {#concern-25}

**Severity**: LOW
**Flagged by**: Agent 7

### What is the problem?

Training scripts set `torch.backends.cudnn.benchmark = False` and `torch.backends.cudnn.deterministic = True` for reproducibility. This is correct for training but should NOT be active during inference.

### Where

All training scripts (04_train*.py):
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Impact

With fixed 512x512 tile input, `cudnn.benchmark = True` would auto-select the fastest convolution algorithm. Disabling it costs 10-20% inference throughput.

### How to fix

Ensure inference scripts (02_generate_masks.py, 11c) set:
```python
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

---

## CONCERN-26: DataLoader num_workers = 0 Starves GPU {#concern-26}

**Severity**: LOW
**Flagged by**: Agents 6, 7

### What is the problem?

In `04_train.py` (line 237):
```python
sys.argv = [sys.argv[0], "-num_workers", "0"]
```

The GPU sits idle while the single CPU thread loads and preprocesses each batch. With v2.0's lazy loading, `num_workers=4-8` would be safe and significantly improve GPU utilization during training.

### How to fix

With the lazy loading already implemented in v2.x scripts, increase num_workers:
```python
sys.argv = [sys.argv[0], "-num_workers", "4"]
```

---

## CONCERN-27: No torch.compile for Inference {#concern-27}

**Severity**: LOW (performance)
**Flagged by**: Agent 7

### What is the problem?

`torch.compile` is not used anywhere in the inference pipeline. On A6000 (Ampere architecture), `torch.compile(model, mode="reduce-overhead")` typically yields 1.5-2x inference speedup.

### How to fix

Add one line before the inference loop in `02_generate_masks.py`:
```python
model = torch.compile(model, mode="reduce-overhead")
```

Note: This requires PyTorch 2.0+ and may need testing with TorchScript models.

---

## CONCERN-28: No Schema Contracts Between Pipeline Stages {#concern-28}

**Severity**: MODERATE
**Flagged by**: Agent 8

### What is the problem?

The parquet schema is implicit. Script 12 assumes `membrane_ring_dab` exists (added by script 11). Script 08 assumes `area_um2` exists (added upstream). If scripts are run out of order, failures are raw `KeyError` exceptions.

### How to fix

Define a schema contract in `src/instanseg_brightfield/schema.py`:
```python
REQUIRED_COLUMNS_STAGE_11 = ["cell_id", "centroid_x", "centroid_y", "polygon_wkb", "dab_mean"]
REQUIRED_COLUMNS_STAGE_12 = REQUIRED_COLUMNS_STAGE_11 + ["membrane_ring_dab", "cldn18_composite_grade"]
```

Validate at the start of each script:
```python
missing = set(REQUIRED_COLUMNS) - set(table.column_names)
if missing:
    raise ValueError(f"Missing columns: {missing}. Run script 11 first.")
```

---

## CONCERN-29: No End-to-End Data Lineage {#concern-29}

**Severity**: MODERATE
**Flagged by**: Agent 8

### What is the problem?

The per-cell parquet files do not record which model version, config hash, or pipeline run produced them. If a bug is found in the membrane measurement, there is no way to determine which results are affected.

### How to fix

Add metadata columns to the parquet output:
```python
table = table.append_column("pipeline_git_sha", pa.array([git_sha] * len(table)))
table = table.append_column("config_hash", pa.array([config_hash] * len(table)))
table = table.append_column("model_version", pa.array(["v1.0"] * len(table)))
```

Or store a sidecar `{slide}_metadata.json` with each parquet.

---

## CONCERN-30: UNCALIBRATED Disclaimer Is Structurally Inadequate {#concern-30}

**Severity**: MODERATE
**Flagged by**: Debate D3 (Clinical Safety Officer)

### What is the problem?

The UNCALIBRATED disclaimer sits in line 1 of `eligibility_summary.json` and line 1 of `eligibility_flips.csv`. Any downstream consumer loading the CSV in pandas will never see it. The CSV also contains patient-level identifiers with "GAINED" labels.

### How to fix

1. Add `_UNCALIBRATED` suffix to all column names that depend on thresholds (e.g., `cldn18_composite_grade_UNCALIBRATED`)
2. Rename output files to include the disclaimer (e.g., `eligibility_flips_UNCALIBRATED.csv`)
3. Add a validation function that prevents loading uncalibrated results without explicit acknowledgment

---

## CONCERN-31: No Docker / Containerization {#concern-31}

**Severity**: LOW (for current stage)
**Flagged by**: Agents 1, 3, 4

### What is the problem?

No Dockerfile, no container definition. The project cannot be deployed or reproduced in an isolated environment.

### How to fix (when ready)

Create a `Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
COPY . /app
WORKDIR /app
RUN pip install -e ".[train]"
```

Not urgent for a research prototype, but required before deployment.

---

## CONCERN-32: No CI/CD Pipeline {#concern-32}

**Severity**: LOW (for current stage)
**Flagged by**: Agents 1, 4

### What is the problem?

No GitHub Actions, no pre-commit hooks, no automated linting. The `ruff` config exists in `pyproject.toml` but there is no evidence it is ever run.

### How to fix

Create `.github/workflows/test.yml`:
```yaml
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: python -m pytest tests/ -v
      - run: ruff check src/ tests/
```

---

## CONCERN-33: Monkey-Patching InstanSeg Internals {#concern-33}

**Severity**: MODERATE (fragility)
**Flagged by**: Agents 1, 6

### What is the problem?

Each training script applies 4+ monkey-patches to the upstream InstanSeg library:
- `_read_images_from_pth` (lazy loading)
- `get_augmentation_dict` (stain augmentation control)
- `_robust_average_precision` (crash prevention)
- The entire `main()` training loop (early stopping)
- `load_model_weights` (custom weight loading)

These create invisible coupling to the exact internal API of `instanseg-torch`. Any upstream update can silently break training.

### How to fix

1. Pin the exact `instanseg-torch` version in requirements
2. Document every patch in a `PATCHING.md` file
3. Add a version check at training script startup:
   ```python
   import instanseg
   assert instanseg.__version__ == "0.1.x", f"Patches validated for 0.1.x, got {instanseg.__version__}"
   ```

---

## CONCERN-34: No Spatial Heterogeneity Analysis {#concern-34}

**Severity**: LOW
**Flagged by**: Agent 9

### What is the problem?

CLDN18.2 expression is known to be highly heterogeneous within tumors (Ruschoff 2023). The pipeline produces a single whole-slide aggregate score without reporting whether the positivity is uniform or concentrated in specific regions.

### Why it matters

A slide with 80% positivity could have uniform 80% staining across all regions (good) or a single hot-spot surrounded by negative tissue (potentially misleading). Spatial heterogeneity affects clinical interpretation.

### How to fix

Add a heterogeneity metric to the eligibility analysis:
- Compute positivity per tile
- Report spatial variance or coefficient of variation
- Flag slides with high spatial heterogeneity for pathologist review

---

## CONCERN-35: Cell Count Discrepancy Between Baseline and Model {#concern-35}

**Severity**: MODERATE (uninvestigated)
**Flagged by**: Agent 2

### What is the problem?

The baseline pipeline detects a mean of 820,594 cells per slide. The new pipeline detects 476,568 cells per slide — a 42% reduction.

### Evidence

From `evaluation/eligibility_analysis/eligibility_summary.json`:
```json
"cell_counts": {
    "baseline_mean": 820594,
    "baseline_median": 798667,
    "cohort_mean": 476568,
    "cohort_median": 354097
}
```

### Possible explanations

1. The baseline counts nuclei (smaller, more numerous) while the model counts cells (larger, fewer per area)
2. The model's quality filters remove more cells
3. The model under-segments touching cells (merges adjacent cells)
4. Different tile sampling strategies

### How to investigate

Compare on a single slide: count cells in the same region with both pipelines and examine the size distributions.

---

## Summary: Concerns by Priority

### Do This Week
| # | Concern | Effort |
|---|---------|--------|
| 07 | Contact McGill tech transfer | 1 day |
| 01 | Get pathologist calibration scores | Start process |
| 02 | Investigate normal mucosa inflation | 1-2 days |
| 04 | Investigate negative correlation | 1-2 days |
| 17 | Create requirements lockfile | 30 min |

### Do This Month
| # | Concern | Effort |
|---|---------|--------|
| 13 | Extract hardcoded paths to config | 2-3 days |
| 15 | Centralize clinical thresholds | 1 day |
| 14 | Consolidate training scripts | 1-2 days |
| 16 | Consolidate compute_hscore() | 1 hour |
| 18 | Fix test suite (install pytest) | 30 min |
| 30 | Improve UNCALIBRATED disclaimers | 1 day |
| 08 | Re-run stain aug ablation (if needed) | 2-3 days |

### Do Before Publication
| # | Concern | Effort |
|---|---------|--------|
| 06 | Multi-reader validation (3 pathologists) | 4-8 weeks |
| 03 | Clarify DL vs bar-filter framing | In manuscript |
| 05 | Address REB "tumour cells" language | 1 day |
| 10 | Investigate cell F1 = 0.24 | 2-3 days |
| 11 | Quantify DAB noise floor on IHC stroma | 1 day |
| 35 | Explain cell count discrepancy | 1-2 days |
| 09 | Add training quality monitoring | 2-3 days |

### Do Before Deployment
| # | Concern | Effort |
|---|---------|--------|
| 24 | Add artifact detection | 2-4 weeks |
| 28 | Schema contracts between stages | 1 week |
| 29 | End-to-end data lineage | 1 week |
| 31 | Docker containerization | 2-3 days |
| 32 | CI/CD pipeline | 1-2 days |
| 19 | Pipeline orchestration (Makefile) | 2-3 days |
| 33 | Document and pin monkey-patches | 1 day |

### Low Priority / Nice to Have
| # | Concern | Effort |
|---|---------|--------|
| 20 | Add experiment tracking (W&B) | 1 day |
| 21 | Fix dataset subsampling (don't overwrite) | 1 hour |
| 22 | Add failed-tile tracking | 1 day |
| 23 | Optimize IoU computation | 1 hour |
| 25 | Enable cudnn.benchmark for inference | 5 min |
| 26 | Increase DataLoader num_workers | 5 min |
| 27 | Add torch.compile | 30 min |
| 34 | Add spatial heterogeneity metric | 1-2 days |
| 12 | Verify training reproducibility (3 seeds) | 3-5 days |
