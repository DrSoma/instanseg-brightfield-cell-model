# Debate Gate 2 Synthesis — Define→Develop Transition
## Date: 2026-03-19 | 6 Agents (5 Claude + Gemini) | Codex 401 Failed

---

## Overall Verdict: UNANIMOUS CAUTION (6/6)

No agent rated GO unconditionally. No agent rated NO-GO. All gave CAUTION with specific conditions.

---

## Agent Summaries

### Google SRE Lead — GO WITH CAUTION
- **Risk 1 (Critical):** Threshold calibration gap — old thresholds (0.10/0.20/0.35) on new membrane-ring metric
- **Risk 2 (Critical):** Zero pathologist validation before integration
- **Risk 3 (High):** GPU 0 SPOF for SegFormer + v2.0 retrain
- **Most likely failure:** Old thresholds on new metric causes misclassification

### Meta ML Infrastructure Lead — CAUTION
- **Risk 1:** Attribution failure from 7 simultaneous changes
- **Risk 2:** w_seed=3.0 + cosine LR + hotstart = dangerous interaction
- **Risk 3:** Lazy loading + num_workers=0 bottleneck
- **Key finding:** `num_workers=1` IS safe with lazy loading (25GB/worker was from eager loading)
- **Key finding:** w_seed=2.0 better than 3.0 (InstanSeg loss divides by 2 for NC mode at line 918)
- **Recommendation:** Split training into Phase A (data) and Phase B (loss/optimizer)

### Apple Biomedical AI Lead — CAUTION / Conditional NO-GO
- **Red flag:** BC_ClassIII inversion (model scores LOWER than pipeline on strong-positive)
- **Biggest risk:** False negatives on strong-positive = denying patients treatment
- **Cannot proceed** without pathologist validation
- **Threshold recalibration:** Use isotonic regression, NOT linear rescaling

### Microsoft Research Principal Scientist — CAUTION
- Training circularity = acceptable (RGB input ≠ DAB labels = knowledge distillation)
- Validation circularity = problematic (all metrics measure DAB at DAB-trained boundaries)
- p<10^-22 is misleading (tiles correlated within slides, need mixed-effects model)
- Adaptive ring 20.6% coverage = selection bias, not fatal
- FM fusion: tile filtering=good, scoring=distraction, PCA embeddings=spatial mismatch
- Minimum: pathologist concordance, clustered stats, one modality-independent check

### Netflix Data Platform Lead — CAUTION, GO with guardrails
- Every Parquet row must be regenerated (different model = different cell boundaries)
- Add `segmentation_model` column for schema versioning
- quality.py `max_cell_nucleus_ratio: 5.0` may need adjustment
- Run v2 into separate output tree, batch migrate 25 slides at a time
- cp -al (hardlink copy) for instant rollback of existing outputs

### Gemini (Senior Biomedical AI Researcher) — CAUTION
- BC_ClassIII reversal is likely fundamental (model under-segments dense DAB)
- 7 simultaneous changes is "scientifically reckless"
- Foundation model fusion = NO-GO for IHC scoring (focus on geometry)
- Need 4-point correlation plot vs pathologist for 10+ slides
- Fix adaptive ring 20.6% success rate before adding complexity

---

## Consensus-Driven Plan Adjustments

### 1. Threshold Management (6/6 agree)
- Do NOT apply old thresholds to membrane-ring DAB for clinical decisions
- Report raw DAB values alongside both old and new thresholds
- Calibrate new thresholds ONLY from pathologist reference data
- Use isotonic regression for the mapping (Apple recommendation)

### 2. Pathologist Validation is a HARD GATE (6/6 agree)
- 30-tile protocol is the minimum
- Must complete BEFORE any clinical deployment or threshold recalibration
- Can proceed with technical validation (pipeline integration, comparison) in parallel
- Cannot proceed with clinical reporting

### 3. Training: 2-Phase Approach (5/6 agree)
- **Phase A (tonight):** lazy loading (78k tiles) + stain aug + disable skeletonize + patience 50 + epochs 300 + num_workers=1
- **Phase B (tomorrow, from Phase A checkpoint):** w_seed=2.0 (not 3.0) + cosine LR + hotstart 25
- If Phase A already fixes skip rate, Phase B may not need w_seed change

### 4. Pipeline Integration: Parallel Run (6/6 agree)
- Output v2 to SEPARATE directory (don't overwrite v1)
- Run BOTH models on same slides for comparison
- Add `segmentation_model` column to Parquet
- Batch migration: 25 slides at a time with sanity checks

### 5. Foundation Model Fusion: Deprioritize (4/6 agree)
- Virchow2 tile filtering = keep (solves epithelial selection problem)
- CONCH/UNI scoring fusion = defer (spatial mismatch, distraction for IHC scoring)
- Fix geometry (adaptive ring, boundary quality) FIRST

---

## Key Technical Insights from Debate

| Finding | Source | Impact |
|---------|--------|--------|
| `num_workers=1` safe with lazy loading | Meta ML | Eliminates I/O bottleneck for 78k tiles |
| w_seed=2.0 not 3.0 (loss/2 for NC mode) | Meta ML | Safer training, undoes NC halving |
| Cosine T_max=100 hard-coded → 3 cycles in 300 epochs | Meta ML | Potential instability, defer to Phase B |
| Validation circularity exists | MSFT Research | Need one modality-independent boundary check |
| p<10^-22 misleading (slide clustering) | MSFT Research | Use mixed-effects model for proper stats |
| quality.py ratio filter may need update | Netflix | max_cell_nucleus_ratio:5.0 for cells vs nuclei |
| BC_ClassIII inversion = measurement geometry | Apple Bio | Membrane ring concentrates/excludes differently |
| Isotonic regression for threshold mapping | Apple Bio | Nonlinearity at high DAB breaks linear correction |
