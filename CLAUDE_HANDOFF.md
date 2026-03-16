# Claude Agent Handoff Prompt

Use the following prompt when starting a new Claude session to continue this project:

---

## Prompt for Next Claude Agent

I'm working on training a brightfield nuclei-and-cells InstanSeg model for whole-slide IHC images. This is a novel project — no such model exists yet.

### Context

We have 200+ CLDN18.2 IHC whole-slide images where DAB chromogen stains the cell membrane (Claudin 18.2 is a membrane protein). The key insight is that the DAB staining itself serves as self-supervised ground truth for cell boundaries — no manual annotation needed.

### What's been done

1. **Orion's InstanSeg pipeline** already runs `brightfield_nuclei` with full polygon extraction, shape measurements (area, perimeter, circularity, solidity, max/min diameter), and intensity measurements on both raw RGB and stain-deconvolved channels (Hematoxylin, DAB, Residual via Beer-Lambert law). Code is at `/home/fernandosoto/Documents/orion-main/orion-ml-worker/orion_ml_worker/wholeslide/processor.py`.

2. **Stain deconvolution** is implemented in both Python (processor.py) and Rust (`orion-main/crates/orion-core/src/analysis/stain_deconv.rs`) using standard H-DAB vectors matching QuPath.

3. **100-slide QuPath benchmark** completed — results in `orion-main/benchmarks/qupath_instanseg_results_100.csv`. Glass test found 7.16M false positives on empty glass across 100 slides.

4. **Bug fix**: CPU futures timeout in processor.py was dropping ~17% of cells in polygon mode. Fixed by increasing timeout from 30s to 300s.

### What needs to be done

The project plan is in `PROJECT_PLAN.md`. The immediate next steps are:

1. **Build the data pipeline** (`scripts/01_extract_tiles.py`, `scripts/02_deconvolve.py`, `scripts/03_generate_cell_masks.py`):
   - Extract tiles from slides at `/pathodata/Claudin18_project/CLDN18_SLIDES_ANON/`
   - Stain deconvolve to isolate DAB channel
   - Threshold DAB to get membrane mask
   - Run InstanSeg brightfield_nuclei for nucleus seeds
   - Watershed with nuclei seeds + membrane boundaries to get cell instance masks

2. **Package training data** into InstanSeg's expected format (`segmentation_dataset.pth`)

3. **Fine-tune InstanSeg** dual-output architecture

### Key files

- This repo: `/home/fernandosoto/Documents/instanseg-brightfield-cell-model/`
- Slides: `/pathodata/Claudin18_project/CLDN18_SLIDES_ANON/` (100 NDPI files)
- Orion processor (reference for stain deconv): `/home/fernandosoto/Documents/orion-main/orion-ml-worker/orion_ml_worker/wholeslide/processor.py`
- InstanSeg package: installed in `/home/fernandosoto/Documents/orion-main/orion-ml-worker/.venv/`
- GPU: 2x NVIDIA RTX A6000 (48GB each)

### Important notes

- The stain vectors for CLDN18 have been calibrated: H=[0.786, 0.593, 0.174], DAB=[0.215, 0.422, 0.881]. Use these instead of defaults for this specific dataset.
- DAB-negative cells have no visible membrane — the pipeline needs to handle this (nucleus expansion fallback or Hematoxylin gradient-based boundaries).
- InstanSeg's TorchScript models cannot be fine-tuned. Raw weights must be requested from the InstanSeg authors at University of Edinburgh (they're responsive on the Image.sc forum).
- Read `README.md` and `PROJECT_PLAN.md` in this repo for full context.

---
