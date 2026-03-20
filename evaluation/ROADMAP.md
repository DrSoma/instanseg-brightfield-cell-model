# CLDN18.2 Brightfield Cell Model — Complete Roadmap
## Date: 2026-03-19 | Status: Day 6 — 5 validation tasks DONE, 200-slide cohort running

---

## Current State (Day 6 update)

- Model segments cell boundaries on brightfield CLDN18.2 IHC using self-supervised DAB membrane ground truth
- Pipeline integration working: 161k cells on 2 BC control slides via production instanseg_batched.py
- 8-test validation confirms DAB selectivity (33x: 162 vs 5 cells/tile on brown vs blue tissue)
- Dr. Fiset validated 97-98% positivity rates as accurate on BC_ClassII and BC_ClassIII
- **NEW: CLSI EP05 PASS** — reproducibility test passes all metrics (CV <15%)
- **NEW: iCAP concordance** — our inter-region H-score diff 1-14 vs baseline 39-71
- **NEW: Morphometric filtering** — standard filter retains 63%, minimal scoring impact
- **NEW: 30-tile pathologist export** ready for Dr. Fiset review
- **NEW: 200-slide cohort running** on GPU 1 (50/200 slides done)
- **NEW: v2.0 Phase A launched** (waiting for GPU 0)
- Composite grading thresholds are PRELIMINARY — need pathologist calibration (3+ population too high at 82%)

---

## A. IMMEDIATE (Can start now)

### A1. Run Full 251-Slide Cohort with v1.0 Model
**What:** Run our validated model through instanseg_batched.py on all 251 CLDN18.2 slides
**Why:** Establishes the clinical baseline — how many patients flip from NOT ELIGIBLE to ELIGIBLE compared to the current nucleus-based pipeline. This is the number that demonstrates clinical impact.
**How:** Use scripts/run_pipeline_comparison.py pattern, adapted for batch processing. GPU 1 is free. ~60 sec/slide = ~4 hours total.
**Input:** 251 slides at /media/fernandosoto/DATA/CLDN18 slides/ + region annotations at /pathodata/Claudin18_project/preprocessing/region_annotations/
**Output:** Per-slide Parquet files with cell boundaries + membrane columns, region summary CSV with H-scores and positivity rates for both old and new model
**Depends on:** Nothing — can start immediately
**Blocks:** Threshold calibration (B2), full cohort analysis, clinical impact assessment
**Automated:** Yes

### A2. Launch v2.0 Phase A Training
**What:** Train improved model using debate-consensus 2-phase approach. Phase A = data/infrastructure changes only.
**Why:** 78k tiles (7.8x more data) + stain augmentation + better convergence parameters should significantly improve model quality.
**How:** `CUDA_VISIBLE_DEVICES=0 nohup python scripts/04_train_v2.py &` — script is written and syntax-checked (819 lines)
**Changes:** Lazy loading (78k tiles), stain augmentation (amount*0.5), disable skeletonize, patience 50, epochs 300, num_workers=1
**NOT changed (Phase B):** w_seed, cosine LR, hotstart epochs
**Input:** data/segmentation_dataset_base.pth (78,511 tiles from 200 slides)
**Output:** models/brightfield_cells_nuclei/model_weights.pth (v2.0 Phase A checkpoint)
**Depends on:** GPU 0 freeing up (SegFormer at epoch 11/30, ~5-6 hours remaining)
**Blocks:** Phase B training, v2.0 validation
**Automated:** Yes — script includes GPU wait loop

### A3. Check SegFormer Results
**What:** Inspect SegFormer membrane detector training output. If val_loss < 0.30, masks are likely better than watershed.
**Why:** Better ground truth masks = better InstanSeg training. SegFormer was trained to detect membrane pixels directly from DAB, producing tighter boundaries than watershed.
**How:** Check /tmp/segformer_stdout.log for final epoch, check models/membrane_segformer/ for checkpoint, run validation on generated masks
**Input:** SegFormer training output (PID 4031189, GPU 0)
**Output:** Validation metrics for SegFormer masks, decision on whether to use for v2.0 retraining
**Depends on:** SegFormer finishing (~5-6 hours)
**Blocks:** Decision on v2.0 ground truth source
**Automated:** Yes

### A4. Prepare 30-Tile Pathologist Export
**What:** Select and export 30 tiles for Dr. Fiset to score manually (0/1+/2+/3+ per cell)
**Why:** This is THE ground truth. All threshold calibration depends on pathologist reference data. Required by REB protocol. Non-negotiable gate before clinical deployment.
**How:** Select tiles stratified by composite score: 10 low-scoring (expected negative/1+), 10 mid-scoring (expected 2+), 10 high-scoring (expected 3+). Export as PNG with cell boundary overlays color-coded by current grade assignment, plus CSV with per-cell measurements (thickness, completeness, intensity, centroid coordinates).
**Input:** Pipeline Parquet output (BC slides or representative patient slides)
**Output:** 30 PNG tiles + 30 CSV files + summary HTML for pathologist review, saved to evaluation/pathologist_export/
**Depends on:** Nothing — can start immediately
**Blocks:** Threshold calibration (B1, B2), formal concordance study (F1)
**Automated:** Yes (tile selection and export). Dr. Fiset scoring is manual (EXTERNAL).

---

## B. THRESHOLD CALIBRATION (Depends on Pathologist Data)

### B1. Collect Dr. Fiset's Per-Cell Scores
**What:** Dr. Fiset reviews 30 exported tiles and assigns 0/1+/2+/3+ grades to each cell
**Why:** Ground truth for threshold calibration. Without this, all grade assignments are preliminary.
**How:** Provide tiles from A4 to Dr. Fiset. He scores using the VENTANA criteria (intensity + completeness + thickness). Record scores in the provided CSV template.
**Input:** 30-tile export from A4
**Output:** CSV with pathologist_grade column filled in for each cell
**Depends on:** A4 (tile export)
**Blocks:** B2 (threshold fitting), F1 (concordance study)
**Automated:** NO — requires Dr. Fiset (EXTERNAL)

### B2. Fit Isotonic Regression for Threshold Calibration
**What:** Map our continuous measurements (completeness × intensity, thickness) to pathologist grades using isotonic regression
**Why:** Converts raw measurements into clinically meaningful 0/1+/2+/3+ scores that match what pathologists see. Isotonic regression preserves monotonicity (higher measurement = higher grade) without assuming linearity.
**How:** Train isotonic regression on pathologist-scored cells from B1. Input features: membrane completeness, membrane intensity (DAB OD), membrane thickness (FWHM). Output: predicted grade. Cross-validate on held-out tiles.
**Input:** Pathologist-scored CSV from B1 + per-cell measurements from pipeline
**Output:** Calibrated threshold function, updated scoring parameters for pipeline
**Depends on:** B1 (pathologist scores)
**Blocks:** Accurate grading on full cohort, publication, clinical deployment
**Automated:** Yes (once pathologist data received)

### B3. Adopt Two-Parameter Continuous Scoring
**What:** Implement completeness × intensity as a continuous score (0-1), following the HER2-CONNECT approach (Visiopharm, FDA-cleared)
**Why:** Continuous scores avoid arbitrary binning. HER2-CONNECT cutoffs (0.12 for 1+/2+, 0.49 for 2+/3+) provide validated reference points. Better for borderline cases.
**How:** Compute per-cell connectivity score = fraction of cell perimeter with positive DAB staining × mean DAB intensity at those positions. Calibrate cutoffs from B1 data.
**Input:** Per-cell completeness and intensity measurements (already computed)
**Output:** Per-cell continuous CLDN18.2 score + calibrated grade cutoffs
**Depends on:** B1 (for calibration), but can compute raw scores now
**Blocks:** Nothing — this is an additional scoring method alongside composite
**Automated:** Yes

---

## C. MODEL IMPROVEMENTS (v2.0 and Beyond)

### C1. v2.0 Phase B Training
**What:** From Phase A checkpoint, add loss/optimizer changes: w_seed=2.0, cosine LR annealing, hotstart 25 epochs
**Why:** Fixes weak seed confidence (logits 1.6 → predicted 4.0+). Edinburgh InstanSeg author confirmed this is the #1 quality bottleneck. w_seed=2.0 (not 3.0) because InstanSeg loss divides by 2 for NC mode (Meta ML debate finding).
**How:** Write 04_train_v2_phaseB.py. Load Phase A checkpoint, modify loss parameters, resume training.
**Input:** Phase A checkpoint from A2
**Output:** v2.0 Phase B model weights
**Depends on:** A2 (Phase A completion)
**Blocks:** v2.0 validation
**Automated:** Yes

### C2. Add Negative Training Examples
**What:** Include stromal/lymphocyte regions with label=background (no cells) during training
**Why:** Model currently detects 5 cells/tile on blue tissue because it was never shown "this is NOT a cell." Adding negative examples teaches the model to suppress detections on non-epithelial tissue.
**How:** For each training tile, identify stromal regions (via Virchow2 clustering or simple DAB thresholding). Set cell_masks=0 in those regions. Retrain.
**Input:** Existing tile dataset + tissue type labels (computed or from Virchow2)
**Output:** Training dataset with explicit negative regions, retrained model
**Depends on:** Nothing (can prepare data now, train after v2.0)
**Blocks:** Improved specificity
**Automated:** Yes

### C3. Retrain with SegFormer Masks
**What:** If SegFormer produces tighter membrane boundaries than watershed, use SegFormer masks as ground truth for InstanSeg training
**Why:** Watershed masks are "loose" — boundaries don't tightly follow actual membrane staining. SegFormer was trained specifically to detect membrane pixels.
**How:** Generate new cell masks using SegFormer membrane predictions + marker-controlled watershed from nuclei seeds. Package into InstanSeg dataset format. Retrain.
**Input:** SegFormer checkpoint from A3, existing nuclei masks
**Output:** Improved cell masks, retrained model
**Depends on:** A3 (SegFormer evaluation)
**Blocks:** Better boundary quality
**Automated:** Yes

### C4. Increase Membrane Coverage Filter
**What:** Raise min_membrane_coverage from 0.3 to 0.5 or 0.7 during mask generation
**Why:** Current 0.3 filter is very permissive — includes cells with minimal membrane signal. Stricter filtering = cleaner ground truth = more confident model = fewer false detections on blue tissue.
**How:** Modify config/default.yaml quality_filter.min_membrane_coverage, regenerate masks, retrain
**Input:** Existing tiles + nuclei masks
**Output:** Filtered dataset, retrained model
**Depends on:** Nothing — can test independently
**Blocks:** Improved selectivity
**Automated:** Yes

---

## D. PIPELINE & SCORING REFINEMENTS

### D1. Two-Model Scoring Approach
**What:** Use fluorescence_nuclei_and_cells (detects ALL cells, 322/tile on blue tissue) for the denominator, our model's membrane DAB for the numerator
**Why:** The VENTANA protocol requires scoring ALL evaluable tumor cells, not just stained ones. Our model preferentially detects stained cells (by design), which inflates positivity %. The fluorescence model provides the true denominator.
**How:** Run both models on each tile. For each fluorescence-detected cell, find the nearest cell in our model's output. If match exists, use our membrane DAB measurement. If no match, score as negative.
**Input:** Both model outputs on same tiles
**Output:** Corrected positivity % = (cells with ≥2+ membrane DAB) / (all fluorescence-detected cells)
**Depends on:** Nothing — both models available
**Blocks:** More accurate clinical scoring
**Automated:** Yes
**Note:** This was the Roche VP debate recommendation

### D2. Virchow2 Epithelial Filtering
**What:** Classify tiles as epithelial vs stromal using Virchow2 embeddings, only score epithelial regions
**Why:** Removes any residual non-epithelial detections. The VENTANA protocol counts only tumor cells.
**How:** Run Virchow2 (already cached, 2.4GB) on all tiles, KMeans K=4-5 (not K=2 per Paige AI debate), validate clusters with morphometrics (nuclear area, eccentricity) NOT DAB (avoids circularity). Feed H-channel to Virchow2, not raw IHC (Paige recommendation).
**Input:** Tiles + Virchow2 model (cached at ~/.cache/huggingface/hub/models--paige-ai--Virchow2)
**Output:** Per-tile epithelial probability, per-cell is_epithelial flag
**Depends on:** Nothing — Virchow2 cached and ready
**Blocks:** Improved tumor cell selection
**Automated:** Yes

### D3. Per-Cell Morphometric Filtering
**What:** Exclude cells with non-epithelial morphology: nuclear area < 80 μm², solidity < 0.9, aspect ratio > 2.0
**Why:** Stromal fibroblasts are elongated (high aspect ratio), lymphocytes are small (low area). Simple geometric criteria can filter them without a foundation model.
**How:** Compute morphometric features from existing polygon_wkb column. Filter cells failing criteria. Recompute H-scores on filtered population.
**Input:** Pipeline Parquet output (polygon_wkb column)
**Output:** Filtered cell population, updated scores
**Depends on:** Nothing — data already available
**Blocks:** Cleaner scoring population
**Automated:** Yes
**Note:** Paige AI CTO debate recommendation

### D4. iCAP Paired Concordance Test
**What:** For all 251 slides, compare model performance on known-positive regions (gastric mucosa) vs known-negative regions (stroma) on the SAME slide
**Why:** Eliminates batch effects. A model with 95% detection on gastric mucosa and 5% on stroma has 19x selectivity. A model with 60% on both has zero selectivity. This is the definitive selectivity metric.
**How:** Use iCAP region annotations (168 GeoJSON files) to stratify cells by region. Compute per-region cell counts, H-scores, positivity %.
**Input:** Full cohort pipeline output from A1 + region annotations
**Output:** Per-slide selectivity ratios, violin plots by region type
**Depends on:** A1 (full cohort run)
**Blocks:** Selectivity validation for publication
**Automated:** Yes
**Note:** Roche VP debate recommendation — "the single most informative test"

### D5. Reproducibility Assessment
**What:** Run same 30 tiles through inference 3x with small perturbations (rotation, crop offset). Must show CV < 15%.
**Why:** Required by CLSI EP05 for software as a medical device. A companion diagnostic must demonstrate analytical precision.
**How:** For each tile, apply 3 random augmentations (±5° rotation, ±10px crop offset). Run model on each variant. Compute coefficient of variation for cell count, H-score, and grade distribution.
**Input:** 30 tiles from A4
**Output:** CV per metric, pass/fail against 15% threshold
**Depends on:** Nothing — can run on current model
**Blocks:** Regulatory compliance, publication
**Automated:** Yes
**Note:** Roche VP debate recommendation

---

## E. HEATMAPS & VISUALIZATION

### E1. Generate Orion Heatmaps for Full Cohort
**What:** Create RGBA PNG heatmap overlays for all 251 slides (composite grade + thickness + density + DAB intensity = 4 per slide = 1004 heatmaps)
**Why:** Enables visual QC by Dr. Fiset across the entire cohort. Can spot problematic slides, staining artifacts, or model failures.
**How:** Extend scripts/generate_orion_heatmap.py for batch processing. Read cell data from Parquet, render at 2560px width.
**Input:** Full cohort Parquet output from A1
**Output:** 1004 PNG heatmaps in evaluation/orion_heatmaps/
**Depends on:** A1 (full cohort run)
**Blocks:** Visual QC
**Automated:** Yes

### E2. Per-Cell Boundary GeoJSON Export for Orion
**What:** Export cell polygons as GeoJSON with properties (grade, thickness, completeness, intensity) for overlay in Orion
**Why:** Allows slide-level review of individual cell boundaries and their assigned grades. More detailed than heatmap overlays.
**How:** Read polygon_wkb from Parquet, convert to GeoJSON FeatureCollection with properties. Color-code by grade.
**Input:** Pipeline Parquet output
**Output:** Per-slide GeoJSON files compatible with Orion annotation system
**Depends on:** Pipeline output existing
**Blocks:** Cell-level visual review
**Automated:** Yes

### E3. Side-by-Side Comparison Viewer
**What:** Generate comparison images showing baseline (brightfield_nuclei) vs our model on the same tissue region
**Why:** Visually demonstrates what the model captures that the baseline misses — membrane boundaries vs nucleus-only detection
**How:** Run both models on same tiles, render overlay with different colors (red=baseline nuclei, green=our cell boundaries)
**Input:** Selected tiles, both models
**Output:** Comparison PNG panels
**Depends on:** Nothing — both models available
**Blocks:** Publication figures
**Automated:** Yes

---

## F. VALIDATION & PUBLICATION

### F1. Formal 30-Tile Concordance Study
**What:** Compute kappa coefficient, ICC (intraclass correlation), Bland-Altman analysis for H-scores between AI and pathologist
**Why:** Quantitative agreement metrics required by REB protocol. Standard for AI in pathology publications. Must demonstrate sufficient agreement before clinical use.
**How:** Compare AI-assigned grades vs Dr. Fiset grades from B1. Compute Cohen's kappa (per-cell), ICC (per-tile H-scores), Bland-Altman plots, sensitivity/specificity for ≥75% at ≥2+ cutoff.
**Input:** Pathologist scores from B1 + AI scores from pipeline
**Output:** Agreement statistics, plots, pass/fail determination
**Depends on:** B1 (pathologist scores), B2 (calibrated thresholds)
**Blocks:** Clinical deployment, publication
**Automated:** Yes (once pathologist data received)

### F2. Multi-Scanner Test
**What:** Run model on slides scanned with Leica Aperio AT2 (in addition to Hamamatsu NanoZoomer)
**Why:** REB protocol specifies both scanner types. Must demonstrate scanner-invariant performance. Stain augmentation in v2.0 should help.
**How:** Obtain Aperio-scanned versions of control slides, run pipeline, compare results
**Input:** Aperio-scanned slides (may need to scan existing blocks)
**Output:** Cross-scanner concordance metrics
**Depends on:** Physical access to Aperio scanner and slides
**Blocks:** Multi-site generalizability claim
**Automated:** NO — requires access to Aperio-scanned slides (EXTERNAL)

### F3. Run on 20-50 Heterogeneous Patient Slides
**What:** Validate that grade distribution spreads across 0/1+/2+/3+ on non-control tissue
**Why:** BC controls have uniformly high expression. Real patient tissue is heterogeneous. Must confirm the model produces clinically meaningful grade distributions on mixed tissue.
**How:** Select 20-50 slides from the 251-slide cohort spanning the H-score range. Run pipeline, examine grade distributions.
**Input:** Subset of full cohort from A1
**Output:** Per-slide grade distributions, comparison with baseline pipeline
**Depends on:** A1 (full cohort run), B2 (calibrated thresholds)
**Blocks:** Confidence in clinical generalizability
**Automated:** Yes

### F4. Write Methods Paper
**What:** Manuscript describing the self-supervised cell boundary learning approach and clinical validation results
**Why:** Novel contribution — no published self-supervised cell boundary learning from IHC membrane staining. The DAB selectivity finding (model preferentially detects expressing cells) is a bonus result.
**How:** Structure: Introduction (CLDN18.2 scoring problem), Methods (self-supervised training, bar-filter validation, composite scoring), Results (8-test validation, pipeline comparison, pathologist concordance), Discussion (clinical implications, limitations).
**Target journals:** Nature Methods, Medical Image Analysis, Modern Pathology
**Input:** All evaluation results, pathologist concordance data, full cohort analysis
**Output:** Manuscript draft
**Depends on:** F1 (concordance study), F3 (heterogeneous validation), A1 (cohort results)
**Blocks:** Publication, academic credit
**Automated:** Yes (draft), NO (revision, submission)

### F5. Consult McGill Tech Transfer
**What:** Discuss patent protection for the self-supervised approach before public disclosure
**Why:** The method (training cell segmentation from IHC membrane staining without annotation) + the implicit epithelial selection finding are potentially patentable. Patent must be filed BEFORE any publication, poster, or preprint.
**How:** Contact McGill Office of Innovation and Partnerships. Prepare invention disclosure form.
**Input:** Technical description of method and results
**Output:** Patent filing decision, invention disclosure
**Depends on:** Nothing — should happen ASAP
**Blocks:** Publication timeline (must file before disclosing)
**Automated:** NO — requires institutional contact (EXTERNAL)

### F6. CLAIM/MINIMAR Reporting
**What:** Prepare reporting documents following CLAIM (Checklist for AI in Medical Imaging) and MINIMAR (Minimum Information for Medical AI Reporting) standards
**Why:** Required by REB protocol. Standard for AI-based biomarker studies. Journals increasingly require these checklists.
**How:** Fill in CLAIM checklist with study design, data, model, evaluation, and clinical validation details. MINIMAR covers minimum reporting standards.
**Input:** All study results and methods
**Output:** Completed checklists for submission
**Depends on:** F1 (concordance study)
**Blocks:** Publication compliance
**Automated:** Yes (draft)

---

## G. RESEARCH EXTENSIONS

### G1. Apply to HER2 IHC
**What:** Train the same self-supervised approach on HER2-stained slides for breast/gastric cancer
**Why:** HER2 is THE canonical membrane biomarker. Same principle: DAB membrane staining → self-supervised cell boundary learning. If it works for CLDN18.2, it should work for HER2 with minimal modification.
**How:** Obtain HER2 IHC slides, extract tiles, generate watershed masks from DAB membrane, train InstanSeg. Existing pipeline works with minimal changes.
**Input:** HER2 IHC slides (need to source)
**Output:** brightfield_cells_nuclei_HER2 model
**Depends on:** Access to HER2 slides
**Blocks:** Multi-marker generalizability claim
**Automated:** Yes (if slides available)

### G2. Apply to PD-L1 IHC
**What:** Adapt for PD-L1 membrane scoring (TPS/CPS)
**Why:** PD-L1 is another membrane biomarker used for immunotherapy eligibility (pembrolizumab, nivolumab). Different biology (immune checkpoint) but same measurement principle (membrane completeness + intensity).
**How:** Similar pipeline. PD-L1 has different scoring criteria (Tumor Proportion Score, Combined Positive Score) but the underlying membrane detection is analogous.
**Input:** PD-L1 IHC slides
**Output:** PD-L1-specific membrane scoring model
**Depends on:** Access to PD-L1 slides, understanding of TPS/CPS criteria
**Blocks:** Broader clinical applicability
**Automated:** Yes (if slides available)

### G3. Foundation Model Integration (CellViT++)
**What:** Use CellViT++ per-cell embeddings for classification refinement
**Why:** CellViT++ extracts rich semantic embeddings per cell at zero additional compute during segmentation. Could improve grade discrimination in borderline cases.
**How:** `pip install cellvit`. Run CellViT++ with UNI encoder alongside our model. Use embeddings as additional features for grade classification.
**Input:** Tiles + CellViT++ model
**Output:** Per-cell semantic embeddings, improved classifier
**Depends on:** Nothing (CellViT++ is open source)
**Blocks:** Research publication on FM integration
**Automated:** Yes
**Priority:** Low (debate consensus: focus on geometry first)

### G4. Multi-Site External Validation
**What:** Run model on slides from another hospital to test generalizability
**Why:** Single-site validation is a known limitation. External validation is required for clinical deployment claims.
**How:** Collaborate with external site, obtain de-identified slides, run pipeline, compare results
**Input:** External slides (need institutional collaboration)
**Output:** Cross-site concordance metrics
**Depends on:** Institutional collaboration agreement
**Blocks:** Generalizability claim, regulatory pathway
**Automated:** NO — requires external collaboration (EXTERNAL)

---

## Dependency Graph (Critical Path)

```
A4 (tile export) → B1 (pathologist scores) → B2 (threshold calibration) → F1 (concordance)
                                                                        → F3 (heterogeneous validation)
                                                                        → F4 (paper)

A1 (251-slide cohort) → D4 (iCAP concordance) → F4 (paper)
                       → E1 (Orion heatmaps)
                       → F3 (heterogeneous validation)

A2 (v2.0 Phase A) → C1 (Phase B) → v2.0 validation
A3 (SegFormer check) → C3 (retrain with SegFormer masks)

F5 (tech transfer) → F4 (paper) [MUST file patent before publication]
```

## What Requires External Help

| Task | Who | What They Do |
|------|-----|-------------|
| B1 | Dr. Fiset | Score 30 tiles (0/1+/2+/3+ per cell) |
| F2 | Lab staff | Scan slides on Leica Aperio AT2 |
| F5 | McGill Tech Transfer | Patent consultation |
| G4 | External collaborator | Provide slides from another hospital |

**Everything else is automated.**
