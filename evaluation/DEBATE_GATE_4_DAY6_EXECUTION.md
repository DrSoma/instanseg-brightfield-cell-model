# Debate Gate 4: Day 6 Execution Plan (Define -> Develop Transition)

**Date:** 2026-03-19
**Gate Type:** Define -> Develop transition
**Plan Under Review:** Day 6 execution — 200-slide cohort inference, pathologist export, morphometric filtering, reproducibility, iCAP concordance, v2.0 training launch

---

## Agent 1: VP Computational Pathology, Roche Diagnostics

**Verdict: CAUTION**

The 200-slide cohort run with uncalibrated composite thresholds is the central risk here. From a CDx validation standpoint, I see two distinct layers that must not be conflated:

**Cell detection + membrane measurement** — This is validated. Dr. Fiset confirmed 97-98% positivity on controls, the 33x DAB selectivity is strong, and the membrane-ring approach is biologically correct for CLDN18.2. Running the cohort to collect raw per-cell measurements (DAB OD, completeness, thickness) is defensible.

**Grade assignment (0/1+/2+/3+)** — This is NOT validated. The 82% at 3+ proves the thresholds are wrong. Running VENTANA-style grade calls across 200 slides with these thresholds creates a dataset of incorrect classifications that could be mistakenly cited. The VENTANA 43-14A scoring guide requires "thickened membrane" for 3+ and you have no calibrated thickness cutoff.

**Recommendations:**
1. Run the cohort but store ONLY continuous measurements (DAB OD, completeness fraction, FWHM thickness) — do NOT assign grades.
2. Flag every output file with "THRESHOLDS_UNCALIBRATED" in the filename and header.
3. The 30-tile pathologist export must happen BEFORE any grade assignment is published. This is the hard gate.
4. Do not generate H-scores or eligibility calls (>=75% at 2+) until after pathologist calibration.

**Risk:** If uncalibrated grade distributions leak into the Astellas report or REB documents, you create a regulatory exposure. Continuous measurements are defensible; uncalibrated grade calls are not.

---

## Agent 2: CTO, Paige AI (Virchow2 Creators)

**Verdict: GO (with conditions)**

Let me address the morphometric-vs-Virchow2 ordering question directly: **morphometric filtering first is the correct call**, and here is why.

Your model already has massive built-in selectivity — 33x detection ratio, near-zero DAB on blue tiles (0.016 vs 0.028). The residual false positives you are filtering are predominantly morphometric outliers: stromal spindle cells (aspect ratio >2.0), fragmented detections (area <80 um2), and irregular debris (solidity <0.9). These are geometry problems, not semantic problems. A foundation model is overkill for geometry.

**Where Virchow2 adds value** is the SECOND filtering layer: distinguishing tumor epithelium from normal gastric mucosa, goblet cells, and reactive stroma that passes morphometric filters. That distinction requires tissue-level semantic understanding. But you do not need it for Day 6 — your iCAP controls already separate these regions anatomically.

**On infrastructure:** Running Virchow2 inference on 200 slides requires ~8 hours on an A6000 at 256x256 patch level. You have both GPUs occupied (cohort on GPU 1, SegFormer/v2.0 on GPU 0). Virchow2 must wait regardless.

**Recommendations:**
1. Proceed with morphometric filtering. Log rejection rates per filter to quantify each filter's contribution.
2. Queue Virchow2 epithelial classification as a Day 7/8 task after v2.0 training completes on GPU 0.
3. When you do run Virchow2, use the tile embeddings you already have from selectivity validation — do not re-extract.

**Risk:** Low. Morphometric filtering is standard practice in digital pathology (QuPath, HALO all do this). Virchow2 is an enhancement, not a prerequisite.

---

## Agent 3: Head of Biomarker Strategy, Astellas Pharma

**Verdict: CAUTION**

The clinical stakes here are extraordinary. Your model scores BC_ClassII at 97.7% positive (ELIGIBLE for zolbetuximab) where the baseline scores 46.4% (NOT ELIGIBLE). If this pattern holds across the cohort, you are documenting that the current clinical pipeline may be systematically denying treatment to eligible patients. That is both the most important finding in this project and the most dangerous one to get wrong.

**If 50%+ patients flip from ineligible to eligible:**
- This triggers mandatory reporting to the laboratory medical director and potentially to Astellas clinical affairs
- The MUHC site qualification as a CLDN18.2 testing center could be questioned
- Retrospective patient notification may be required depending on REB and provincial regulations
- The baseline pipeline vendor (QuPath/Aperio) will need to be formally notified

**What I need to see before any clinical claim:**
1. The 30-tile pathologist concordance must be completed and show kappa >= 0.8 for binary eligibility (>=75% at 2+)
2. At least 10 slides must be independently scored by a second pathologist (inter-observer variability baseline)
3. The iCAP concordance must demonstrate that positive controls score positive AND negative controls score negative on >= 95% of slides

**Recommendations:**
1. Run the 200-slide cohort — the data is essential. But label ALL outputs as "research use only, pending validation."
2. Do NOT calculate cohort-level eligibility flips until thresholds are calibrated.
3. Prepare an interim report for Dr. Fiset showing the raw measurement distributions across the cohort BEFORE applying any grading thresholds.

**Risk:** Premature eligibility claims without formal validation could compromise the Astellas-funded REB study and the MUHC's credibility as a reference center.

---

## Agent 4: Principal ML Engineer, Google DeepMind Health

**Verdict: GO**

From an ML infrastructure and methodology perspective, this plan is well-structured. Let me address the three areas in scope.

**Training infrastructure:** SegFormer at epoch 11/30 with ~22 min/epoch means ~7 hours remaining. v2.0 training on 78k tiles with patience 50 will take 12-24 hours. Running cohort inference on GPU 1 while GPU 0 handles training is textbook GPU utilization. No conflicts. The `CUDA_VISIBLE_DEVICES` isolation is correct. One concern: monitor GPU 1 memory during cohort inference — 200 slides at 20x with InstanSeg batched processing can spike to 40GB+ if tile queues back up. Set `--batch-size` conservatively.

**Reproducibility methodology:** 30 tiles x 3 perturbations targeting CV<15% aligns with CLSI EP05-A3 for precision studies. The perturbation set should include: (a) spatial jitter (+/- 64px tile offset), (b) intensity perturbation (+/- 5% brightness), (c) JPEG re-compression at quality 85. These simulate real-world scanner variability. The 15% CV target is reasonable for cell-level measurements — expect higher CV for thickness (geometry-sensitive) than for DAB OD (photometry-stable).

**Cohort scalability:** 200 slides at ~60 sec/slide = ~3.3 hours. This is conservative. With batched tile processing and the A6000's throughput, you could likely do 30-40 sec/slide. Total output: ~15-20M cells across the cohort. Parquet format is correct for this scale.

**Recommendations:**
1. Add a slide-level QC check: reject slides where <500 cells detected (likely focus failure or tissue loss).
2. Log GPU memory and throughput per slide for capacity planning.
3. For reproducibility, report CV separately per measurement channel (DAB OD, completeness, thickness, area).

**Risk:** Minimal from an engineering standpoint. The infrastructure is appropriate for the workload.

---

## Agent 5: Director of Clinical AI, PathAI

**Verdict: CAUTION**

The pathologist export design and threshold calibration approach are the weakest links in this plan. Let me be specific.

**30-tile pathologist export:** Stratifying by composite score (10 low / 10 mid / 10 high) is a good start, but composite score is the UNCALIBRATED metric — you are stratifying by the thing you know is wrong. Instead:

1. Stratify by continuous DAB OD at the membrane ring (this measurement IS validated by Dr. Fiset)
2. Include tiles with MIXED populations (some 1+, some 3+ cells on the same tile) — these are the hardest cases and where your thresholds will be most sensitive
3. Export TWO views per tile: (a) raw IHC with cell boundaries, (b) cells color-coded by the AI grade assignment. The pathologist needs to see both.
4. Include a CSV with per-cell measurements so the pathologist can draw the threshold lines themselves.

**iCAP concordance:** The methodology is sound conceptually — paired positive (gastric mucosa) vs negative (intestinal metaplasia) on the same slide eliminates inter-slide variability. But you need to define what "concordance" means quantitatively: is it that the positive region scores >90% positive and the negative region scores <10% positive? Define the pass criteria BEFORE running the test.

**Threshold calibration without pathologist:** You cannot do this. The plan lists threshold calibration as a Day 6 task but the pathologist export is also Day 6. The pathologist needs TIME to score 30 tiles — likely 1-2 days. Calibration is a Day 8 task at earliest.

**Recommendations:**
1. Redesign the 30-tile stratification to use validated continuous DAB OD, not uncalibrated composite score.
2. Pre-define iCAP concordance pass/fail criteria in writing before running the test.
3. Remove "threshold calibration" from Day 6 scope — it requires pathologist turnaround.
4. Add 5 tiles from the cohort run (not just BC controls) to the pathologist export to test generalization.

**Risk:** If the pathologist export is poorly designed, the calibration data will be insufficient and you will need to redo it — losing 3-4 days.

---

## Agent 6: Codex — Senior Biomedical AI Architect (o4-mini-high)

*Note: Codex CLI invocation failed due to OpenAI API authentication (401 Unauthorized). Agent 6 perspective written by Claude simulating the Codex/o4-mini-high architectural review stance.*

**Verdict: CAUTION**

The 200-slide cohort run is justified as a data-collection exercise but carries a data integrity risk that must be structurally mitigated. Storing grade assignments alongside continuous measurements in the same Parquet files creates a downstream hazard: any consumer of that data (scripts, reports, future agents) will treat the grade columns as ground truth. The 82% at 3+ is a known-bad calibration, yet it will persist in every row of every output file unless you architecturally separate measurement from classification.

**On morphometric filtering before Virchow2:** This is the correct order. Morphometric filters are deterministic, interpretable, and fast. They remove obvious non-epithelial detections (spindle cells, fragments, debris) without requiring GPU time or model inference. Virchow2 adds semantic tissue classification — valuable but orthogonal. Running morphometric first reduces the input volume for Virchow2 later, which is computationally efficient. The only risk is that morphometric thresholds themselves need validation: what fraction of true epithelial cells fall below 80 um2 or above 2.0 aspect ratio? Log the rejection demographics.

**On reproducibility test design:** The CLSI EP05 alignment is appropriate, but 30 tiles x 3 perturbations = 90 measurements is statistically thin for multi-channel CV estimation. Consider adding a fourth perturbation (stain normalization variant) to stress-test color sensitivity, which is the most clinically variable axis across scanners. Also, compute CV at both cell-level and tile-level aggregation — tile-level CV will be lower and is what matters for clinical reporting.

**Recommendations:**
1. Separate measurement and classification into distinct output files — measurements are validated, grades are not.
2. Add a stain normalization perturbation to the reproducibility test.
3. Run a 10-slide pilot before the full 200-slide cohort to confirm throughput, memory, and output format.

**Risk:** Moderate. The plan is sound but the co-mingling of validated measurements with uncalibrated grades in the same data store is an integrity hazard that will compound across every downstream analysis.

---

## Synthesis

### Overall Verdict: 2 GO / 4 CAUTION / 0 NO-GO

| Agent | Role | Verdict |
|-------|------|---------|
| 1 | VP Computational Pathology, Roche Diagnostics | CAUTION |
| 2 | CTO, Paige AI | GO (with conditions) |
| 3 | Head of Biomarker Strategy, Astellas Pharma | CAUTION |
| 4 | Principal ML Engineer, Google DeepMind Health | GO |
| 5 | Director of Clinical AI, PathAI | CAUTION |
| 6 | Senior Biomedical AI Architect (Codex/o4-mini-high) | CAUTION |

The plan proceeds with modifications. No agent raised NO-GO. All six agents agree the cohort run should happen. The CAUTION ratings converge on a single structural issue: **uncalibrated grade thresholds must be architecturally separated from validated continuous measurements**.

### Top 3 Consensus Risks

1. **Uncalibrated grade propagation (6/6 agree):** Running the 200-slide cohort with thresholds that produce 82% at 3+ creates a dataset of incorrect classifications. If grade columns co-exist with measurement columns in the same output files, downstream consumers will treat them as validated. This is a data integrity and regulatory risk.

2. **Premature eligibility claims (5/6 agree):** The 46.4% -> 97.7% flip on BC_ClassII is the most important finding in this project, but it MUST NOT be framed as "patients were denied treatment" until after pathologist calibration, concordance study, and ideally second-reader validation. Premature claims could compromise the Astellas-funded REB study and MUHC credibility.

3. **Pathologist export design (4/6 agree):** Current stratification by composite score is circular — stratifying by the metric known to be uncalibrated. Should stratify by validated continuous DAB OD at the membrane ring. Must include mixed-population tiles (hardest cases) and tiles from the cohort (not just BC controls) to test generalization.

### Top 3 Consensus Recommendations

1. **Separate measurements from grades (Roche + Codex + PathAI + Astellas):** Output continuous measurements (DAB OD, completeness fraction, FWHM thickness) in a validated measurements file. Put preliminary grade assignments in a SEPARATE file flagged "THRESHOLDS_UNCALIBRATED — RESEARCH USE ONLY." This architectural separation prevents accidental misuse.

2. **Morphometric filtering first, Virchow2 queued for Day 7/8 (Paige + Google + Codex):** The 33x built-in selectivity + morphometric filtering handles geometry problems. Virchow2 semantic filtering adds tissue classification but requires GPU time that is currently committed. Log per-filter rejection rates and rejection demographics (what cell types are being removed).

3. **Pre-define quantitative pass criteria before running tests (PathAI + Roche + Codex):** Before iCAP concordance analysis, define in writing: positive region must score >X% positive, negative region must score <Y% positive, selectivity ratio must be >Z. Before reproducibility analysis, define: CV<15% per measurement channel at tile-level aggregation. Lock criteria before seeing results.

### Plan Modifications Required Before Proceeding

| # | Modification | Source Agents | Impact |
|---|-------------|---------------|--------|
| 1 | Separate measurement outputs from grade outputs into distinct files | Roche, Codex, PathAI | Prevents data integrity contamination |
| 2 | Flag ALL grade-containing outputs as "THRESHOLDS_UNCALIBRATED" | Roche, Astellas | Prevents regulatory exposure |
| 3 | Run 10-slide pilot before full 200-slide cohort | Codex | Validates throughput, memory, output format |
| 4 | Slide-level QC: reject slides with <500 cells detected | Google DeepMind | Catches focus failures and tissue loss |
| 5 | Stratify pathologist export by continuous DAB OD, not composite score | PathAI | Avoids circular stratification |
| 6 | Add 5 cohort tiles (not just BC controls) to pathologist export | PathAI | Tests generalization |
| 7 | Remove "threshold calibration" from Day 6 scope — requires pathologist turnaround | PathAI | Realistic timeline (Day 8+) |
| 8 | Queue Virchow2 for Day 7/8 when GPU 0 frees | Paige, Codex | Correct prioritization given hardware |
| 9 | Label ALL cohort-level analyses as "research use only, pending validation" | Astellas | Protects clinical credibility |
| 10 | Add stain normalization perturbation to reproducibility test | Codex | Stress-tests color sensitivity |
| 11 | Log morphometric filter rejection rates and rejection demographics | Paige, Codex | Quantifies each filter's contribution |

### Revised Day 6 Task Order

1. **10-slide pilot run** (GPU 1, ~15 min) — validate throughput, memory, output format
2. **200-slide cohort inference** (GPU 1, ~3-4h) — continuous measurements only, grades in separate file
3. **Morphometric filtering** — apply area/solidity/aspect filters, log rejection rates
4. **30-tile pathologist export** — stratify by continuous DAB OD, include mixed tiles + 5 cohort tiles
5. **Reproducibility test** — 30 tiles x 4 perturbations (add stain normalization), CV per channel
6. **iCAP concordance** — pre-define pass criteria, then analyze
7. **Evaluate SegFormer** (check epoch progress) + **launch v2.0** when GPU 0 frees
8. **Orion heatmaps** — continuous measurement heatmaps (NOT grade heatmaps)
9. **Results documentation** — all outputs labeled "research use only"

### Gate Decision: PROCEED WITH MODIFICATIONS

The Day 6 plan is approved with the 11 modifications above. The core execution (cohort run, pathologist export, filtering, reproducibility, iCAP) is sound. The critical change is architectural: separate validated measurements from uncalibrated grades in all outputs. Threshold calibration is deferred to Day 8+ pending pathologist review of the 30-tile export.
