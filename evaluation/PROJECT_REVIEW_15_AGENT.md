# Objective Project Review: instanseg-brightfield-cell-model

**Date**: 2026-03-21
**Method**: 15-agent Double Diamond workflow (9 investigative + 5 debate + 1 quality gate)
**Scope**: Full system — code, architecture, ML, science, clinical readiness, data engineering

---

## Final Score: 5.5–6.0 / 10 (stage-adjusted for 7-day sprint)

The range reflects genuine disagreement. Defense argued 7.5 (context matters); prosecution argued 3.5 (core science unvalidated); quality gate settled at 5.5–6.0.

---

## Scoreboard

| Dimension | Score | Assessor |
|-----------|-------|----------|
| Code Quality | 5.9/10 | Code Reviewer |
| Scientific Rigor | 5.5/10 | Research Synthesizer |
| Clinical Readiness | 4.0/10 | Strategy Analyst |
| Architecture | 6.0/10 | Backend Architect |
| Maturity | Alpha (strong) | Product Writer |
| ML Engineering | 5.5/10 | Senior ML Engineer |
| Performance | 6.0/10 | Performance Engineer |
| Data Engineering | 5.0/10 | Data Engineer |
| CompPath Domain Fit | 6.0/10 | CompPath AI Specialist |
| Innovation Value | 8.5/10 | Consensus |

---

## The Single Most Important Finding

**The deep learning model does not trace membranes. The bar-filter does.**

The fixed-ring compartment test FAILS (membrane DAB 0.254 < cytoplasm DAB 0.274). The Aperio bar-filter post-processor PASSES (gap +0.127). This means the CLDN18.2 scoring pipeline is best described as **"DL-assisted classical membrane scoring"** — the neural network provides cell masks via watershed from nuclei seeds, and classical image processing (oriented bar-filters) performs the actual membrane measurement.

This is not a flaw per se — bar-filters have FDA precedent (Aperio HER2). But it means the system's intellectual contribution is the **self-supervised training + bar-filter pipeline integration**, not "deep learning membrane segmentation."

---

## Top Strengths (consensus across 9+ agents)

1. **Genuinely novel method** — zero-annotation self-supervised membrane IHC scoring has no published precedent and is potentially patentable
2. **Production-scale execution** — 84.3M cells, 191 slides, crash-resilient pipeline with config hashing and atomic writes
3. **Unprecedented self-criticism** — 6 adversarial debate gates with 60+ agent-perspectives exceeds most published peer review
4. **Pathologist-validated output** — Dr. Fiset confirmed 97-98% positivity as accurate
5. **Clinical throughput feasible today** — 19 slides/hr on 2x A6000, optimizable to ~50/hr
6. **Slide-level data splitting** — correctly prevents tile-level data leakage between train/val/test
7. **CLSI EP05 reproducibility PASS** — appropriate clinical laboratory standard
8. **Well-designed core library** — `src/instanseg_brightfield/` has clean separation of concerns, type hints, Google-style docstrings

---

## Top Risks (consensus across 5+ agents)

1. **Uncalibrated thresholds** — 82% at 3+ on controls means scoring is non-discriminating. All eligibility numbers contingent on calibration.
2. **No tumor cell classification** — VENTANA scoring requires "evaluable tumor cells" only. Pipeline scores all DAB-positive cells.
3. **Single-site, single-reader, single-scanner** — zero generalizability evidence.
4. **IP unprotected** — McGill tech transfer not contacted; novelty window at risk with any disclosure.
5. **Cohort data in /tmp/** — 47.7M cells of GPU-computed data lost on reboot.
6. **Cell F1 = 0.24** — model disagrees with its own training signal 76% of the time at cell level.
7. **Training has zero F1 feedback** — `_safe_ap` silences all metric exceptions; early stopping on loss only.
8. **40+ hardcoded paths** — project non-portable to another machine.
9. **No experiment tracking** — W&B/MLflow optional deps installed but never used.
10. **Stain augmentation ablation invalid** — confounded by sigma collapse; finding should be retracted.

---

## Detailed Dimension Reports

### 1. Code Quality (5.9/10)

**Strengths:**
- Clean `src/instanseg_brightfield/` library (970 lines, 8 modules) with proper type annotations and docstrings
- Centralized `config/default.yaml` as single source of truth
- Numbered script convention (00-14) provides clear execution order
- Consistent logging with `logging.getLogger(__name__)`

**Weaknesses:**
- 4 near-identical training scripts (04_train_v2*.py) sharing ~95% code — should be 1 parametric script
- 5+ monkey-patches of InstanSeg internals create fragile upstream coupling
- 40+ hardcoded absolute paths to `/home/fernandosoto/...`
- Test suite exists (622 lines) but cannot run (pytest not installed)
- Zero test coverage on the 46 scripts (28K lines)
- Clinical thresholds scattered across 4+ files with no single authoritative source

**Top Recommendations:**
1. Consolidate 4 training scripts into 1 with `--stain-aug` flag
2. Extract all paths into `config/default.yaml` or `config/paths.yaml`
3. Create `config/clinical_thresholds.yaml` centralizing all clinical cutoffs
4. Generate `pip freeze > requirements-lock.txt`
5. Fix pytest installation and add CI

### 2. Scientific Rigor (5.5/10)

**Strengths:**
- Multi-pronged validation: boundary validation, CLSI EP05 reproducibility, iCAP concordance, morphometric filtering, multi-architecture comparison
- Data leakage prevention via slide-level splitting with deterministic seeding
- Sigma collapse root cause analysis is one of the most thorough training failure investigations in a research project
- Explicit "UNCALIBRATED" disclaimers show scientific caution

**Weaknesses:**
- No confidence intervals reported anywhere (point estimates only across all results)
- Negative correlation with baseline (r=-0.32 for %2+) — likely a threshold mismatch artifact but uninvestigated
- Single pathologist reviewer (standard is 3+ independent readers with kappa >= 0.6)
- No independent ground truth for cell segmentation (model validated indirectly via clinical output)
- 38.5% tile skip rate in boundary validation unexplained
- Cell count discrepancy (baseline 820k vs model 476k per slide) not discussed
- Stain augmentation ablation invalidated by sigma collapse confound

**Top Recommendations:**
1. Investigate negative correlation — run on matched slides with matched thresholds
2. Add bootstrap 95% CIs to all reported metrics
3. Run 05_evaluate.py and report standard segmentation metrics
4. Explain cell count discrepancy between baseline and model
5. Retract or qualify the stain augmentation ablation finding

### 3. Clinical Readiness (4.0/10)

**Strengths:**
- No commercial AI solution exists for CLDN18.2 scoring — genuine market gap
- VENTANA companion diagnostic (FDA Oct 2024) uses manual scoring — automation opportunity
- 30-tile pathologist export prepared for calibration
- CLSI EP05 reproducibility passes clinical laboratory standard

**Blocking Gaps:**
- Uncalibrated thresholds (30.4% eligibility vs 0.9% baseline)
- Single-site, single-reader validation
- No Software Requirements Specification (SRS)
- No ISO 14971 risk analysis
- No CLAIM/MINIMAR checklists
- Staining protocol alignment with VENTANA reference undocumented
- No cross-scanner validation (Hamamatsu only)
- No artifact exclusion (folds, pen marks, necrosis)

**Regulatory Path:**
- Health Canada: SaMD Class II (QA tool) or Class III (treatment decision)
- FDA: No predicate for AI CLDN18.2 scoring; CDx requires PMA
- Self-supervised training approach is genuinely novel — no regulatory precedent

**Gap Analysis:** 10 must-haves before clinical use, 5 should-haves before publication, 5 nice-to-haves for commercialization (see full Strategy Analyst report).

### 4. Architecture (6.0/10)

**Strengths:**
- Clear numbered pipeline (00-14) with well-defined stage responsibilities
- PipelineManifest provides crash-resilient, idempotent execution with atomic writes
- Multi-GPU inference with round-robin distribution and VRAM-aware batch sizing
- CUDA OOM recovery with automatic batch-size halving
- Distribution drift monitoring flags anomalous slides

**Weaknesses:**
- No orchestration (no Makefile, Snakemake, or pipeline runner)
- Clinical scripts (07-14) significantly less disciplined than core pipeline (00-06)
- Hardcoded paths break portability
- No CI/CD, no Docker, no deployment automation
- No config validation schema
- `compute_hscore()` duplicated verbatim across scripts instead of in library

**Top 3 Architectural Risks:**
1. Configuration drift between core pipeline and clinical scripts
2. No orchestration — pipeline is a manual sequence of scripts
3. Hardcoded external paths break reproducibility

### 5. ML Engineering (5.5/10)

**Strengths:**
- Slide-level data splitting prevents leakage (8/10)
- Comprehensive seed setting (Python, NumPy, PyTorch, CUDA, cuDNN)
- Multiple architectures explored (CellPose, StarDist, BiomedParse, SegFormer)
- Clinical integration strong (QuPath export, eligibility analysis)

**Weaknesses:**
- `_safe_ap` wrapper silences ALL metric exceptions — training has zero F1 feedback
- Cell F1 = 0.24 against self-supervised ground truth
- No experiment tracking despite W&B being an optional dependency
- Stain augmentation ablation invalidated by sigma collapse
- `num_workers=0` starves GPU during training
- Deterministic mode unnecessarily slows inference
- 20% training success rate (5 attempts, 1 success)

**Critical ML Risks:**
1. Training loop flies blind on segmentation quality
2. Self-supervised labels amplify teacher model biases
3. Heavy monkey-patching creates untested interaction surfaces
4. Cell boundaries don't trace membranes — bar-filter does
5. No ablation studies with controlled variables

### 6. Performance (6.0/10)

**Current Throughput:** ~19 slides/hr (mixed workload, dual A6000)
**Optimized Estimate:** ~50-60 slides/hr with torch.compile + CUDA streams + adaptive giant-slide handling

**What's Good:**
- AMP inference correctly implemented
- Multi-GPU with model replicas and round-robin
- Memory backpressure prevents OOM
- VRAM-aware batch sizing well-calibrated
- Evolution from 8 to 34 slides/hr shows active iteration

**Missing Optimizations:**
- No `torch.compile` (1.5-2x potential speedup)
- No CUDA streams for overlapping transfer + compute
- cuDNN benchmark disabled during inference
- No async I/O prefetch in main pipeline
- No TensorRT/ONNX Runtime
- Giant slide handling is brute-force

**Clinical Feasibility:** 50 slides/day feasible today; 200/day needs optimization.

### 7. Data Engineering (5.0/10)

**Strengths:**
- Parquet with zstd compression — excellent for 84M+ cells
- PipelineManifest with config hashing and atomic writes
- Quality filtering with distribution drift monitoring

**Weaknesses:**
- Critical cohort data in `/tmp/` (volatile storage)
- No formal schema definition for parquet data model
- No end-to-end data lineage (cell → model version → config → slide)
- No schema validation between pipeline stages
- Membrane columns added in-place with no version tracking
- No patient demographics, scanner info, or stain batch metadata
- No data versioning or backup strategy

### 8. CompPath Domain Fit (6.0/10)

**Strengths:**
- Bar-filter membrane detection has FDA precedent (Aperio HER2)
- Multi-dimensional scoring (intensity + completeness + thickness) mirrors VENTANA criteria
- DAB deconvolution correctly implemented with calibrated stain vectors
- Membrane thickness (10px FWHM) is biologically plausible

**Critical Gaps:**
1. No tumor cell classification — scores all cells, not just evaluable tumor
2. Fixed stain vectors with H&E cross-reactivity (0.363 OD "DAB" on tissue with zero DAB)
3. No artifact exclusion (folds, pen marks, necrosis inflate scores)
4. No spatial heterogeneity analysis
5. No apical vs. basolateral membrane distinction

---

## Debate Gate Summary

### Gate 1: Define → Develop (5 agents)

| Debater | Position | Score |
|---------|----------|-------|
| Defense Advocate | Context matters, 7-day sprint is exceptional | 7.5/10 |
| Prosecution Advocate | Core science unvalidated, bar-filter not DL | 3.5/10 |
| Clinical Safety Officer | Patient safety gaps underweighted | 4.0/10 |
| VC Technology Assessor | Innovation doesn't rescue unvalidated science | 5.5/10 |
| Meta-Reviewer | Reviewers applied wrong standards, misread data | >5.5/10 |

**Key Corrections:**
- r=-0.32 is likely a threshold mismatch artifact (different compartments, different cell counts)
- Boundary validation was partially misread — bar-filter PASSES
- Production standards applied to Day 7 research sprint = category error
- BUT: DL model genuinely doesn't trace membranes — bar-filter does
- Publishing patient-level eligibility with CSV-line-1 disclaimer is structurally inadequate

### Gate 2: Develop → Deliver (1 agent)

**Adjustments:**
- Priority #1 should be moving data from /tmp/ (catastrophic prevention), not tech transfer
- System should be labeled "DL-assisted classical membrane scoring" in any write-up
- 50% to publishable is accurate but remaining work includes validation studies

---

## Corrected Top 5 Priorities

| # | Action | Why | Effort |
|---|--------|-----|--------|
| 1 | **Move data out of /tmp/** | Prevents catastrophic, irreversible data loss | 1 hour |
| 2 | **Contact McGill tech transfer** | Novelty window closes with any disclosure | 1 day |
| 3 | **Get Dr. Fiset's calibration scores** | Everything downstream depends on this | 2-4 weeks |
| 4 | **Investigate negative correlation** | Is it threshold mismatch or real? | 1-2 days |
| 5 | **Create requirements lockfile** | Minimum reproducibility barrier | 30 minutes |

---

## Stage Assessment

| Component | Stage | % to Publishable | % to Deployable |
|-----------|-------|-------------------|-----------------|
| Cell detection | Beta | 70% | 30% |
| Scoring pipeline | Prototype | 40% | 10% |
| Training pipeline | Alpha | 50% | 15% |
| Documentation | Alpha | 60% | 15% |
| Reproducibility | Prototype | 30% | 10% |
| Clinical validation | Prototype | 25% | 5% |
| Regulatory | Not started | 5% | 2% |
| **Overall** | **Alpha (strong)** | **~45%** | **~15%** |

---

## The Honest Bottom Line

This project accomplished in 7 days what most computational pathology labs spend 3-6 months on. The self-supervised approach is a genuine intellectual contribution worth protecting. The engineering demonstrates real sophistication in GPU pipeline design and crash resilience.

But the project sits at a critical inflection point. The method's validity hinges on one key experiment — pathologist threshold calibration — that hasn't happened yet. Until then, the "16 patients gained eligibility" headline is directionally compelling but scientifically premature.

**Reframe the narrative**: from "DL membrane scoring that changes patient eligibility" to "DL-assisted classical membrane measurement that produces systematically different results from nuclear-expansion scoring, pending calibration against pathologist consensus."

That framing is more defensible, more publishable, and more honest.

---

*Generated by 15 Claude agents via Double Diamond workflow with 2 debate gates*
*Total agent-perspectives: 9 investigative + 5 debate + 1 quality gate = 15*
