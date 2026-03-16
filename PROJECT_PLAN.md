# Project Plan: Brightfield Cell+Nucleus InstanSeg Model

## Phase 1: Data Pipeline (auto-generate cell boundary masks)

### Step 1.1: Tile Extraction
- Extract 256×256 or 512×512 tiles from tissue regions of CLDN18.2 slides
- Use Orion's existing tissue masking to skip background
- Target: ~10,000 tiles from 20-30 slides (variety of staining intensity and tissue density)

### Step 1.2: Stain Deconvolution
- Deconvolve each tile into Hematoxylin, DAB, Residual channels
- Use calibrated H-DAB vectors from the CLDN18 pipeline
- Output: DAB concentration map per tile (float32)

### Step 1.3: DAB Membrane Mask
- Threshold DAB channel to extract membrane outlines
- Apply morphological operations (closing to connect gaps, thinning to get 1-pixel-wide boundaries)
- Handle edge cases: weak staining, over-staining, tissue folds

### Step 1.4: Nucleus Detection
- Run `brightfield_nuclei` InstanSeg on each tile
- Output: nucleus instance mask (int32 labeled array)

### Step 1.5: Watershed Cell Segmentation
- Seeds: nucleus centroids from Step 1.4
- Boundaries: DAB membrane mask from Step 1.3
- Algorithm: marker-controlled watershed on the inverted DAB distance transform
- Output: cell instance mask (int32 labeled array)

### Step 1.6: Quality Control
- Filter out cells where:
  - No membrane signal detected around nucleus
  - Cell area is unreasonably large (>5x nucleus area for epithelial cells)
  - Cell boundary doesn't form a closed polygon
- Pathologist spot-check: review ~100 random tiles to validate auto-generated masks

## Phase 2: Training Data Preparation

### Step 2.1: Format Conversion
- Convert paired nucleus/cell masks into InstanSeg's `segmentation_dataset.pth` format
- Follow the `load_datasets.ipynb` template from the InstanSeg repo

### Step 2.2: Train/Val/Test Split
- Split by slide (not by tile) to avoid data leakage
- 70% train / 15% validation / 15% test
- Stratify by staining intensity (low/medium/high DAB)

### Step 2.3: Augmentation Strategy
- Random rotation, flip, color jitter
- Stain augmentation (vary stain vectors slightly to simulate batch-to-batch variation)
- Random crop within tiles

## Phase 3: Model Training

### Step 3.1: Architecture
- Use InstanSeg dual-output architecture (same as `fluorescence_nuclei_and_cells`)
- Two output heads: nucleus embeddings + cell embeddings
- Start from `brightfield_nuclei` pre-trained weights for the shared encoder

### Step 3.2: Fine-tuning
- Contact InstanSeg authors for raw model weights (not TorchScript)
- Include CPDMI dataset to prevent catastrophic forgetting (per InstanSeg team recommendation)
- Learning rate: low (1e-5 to 1e-4) since we're fine-tuning, not training from scratch
- Monitor: loss convergence, mAP on validation set

### Step 3.3: Hyperparameter Tuning
- DAB threshold sensitivity analysis
- Watershed parameters (compactness, distance transform sigma)
- Augmentation intensity
- Batch size / learning rate schedule

## Phase 4: Evaluation

### Step 4.1: Metrics
- mAP (COCO-style) for both nucleus and cell detection
- F1 / Precision / Recall at IoU thresholds 0.5 and 0.75
- Dice coefficient for segmentation quality
- Nucleus/Cell area ratio distribution (biological plausibility check)

### Step 4.2: Baselines
- **Nucleus expansion** (QuPath's default approach): expand nuclei by 5 µm fixed radius
- **Nucleus only**: `brightfield_nuclei` without cell boundaries
- **Fluorescence model on brightfield**: `fluorescence_nuclei_and_cells` (expected to fail — domain mismatch)

### Step 4.3: Pathologist Validation
- Select 50 test tiles, show predicted cell boundaries overlaid on the slide
- Have pathologist rate: Correct / Acceptable / Wrong for each cell
- Compute inter-rater agreement if multiple pathologists available

## Phase 5: Export and Integration

### Step 5.1: Model Export
- Export to TorchScript format via InstanSeg's `export_model.ipynb`
- Verify compatibility with QuPath's InstanSeg extension
- Verify compatibility with Orion's Python InstanSeg adapter

### Step 5.2: Orion Integration
- Add model to Orion's model registry
- Update `_process_tile_labels_with_polygons()` to handle dual nucleus+cell masks
- Compute compartment measurements: Nucleus, Cell, Cytoplasm (Cell−Nucleus), Membrane
- Add `nucleus_cell_ratio` shape measurement

### Step 5.3: Benchmarking
- Re-run the 100-slide benchmark with the new model
- Compare: cell counts, compartment measurements, speed, resource usage
- Update the board presentation HTML report

## Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|-------------|
| Phase 1: Data Pipeline | 1-2 weeks | Slides, GPU access |
| Phase 2: Training Data | 2-3 days | Phase 1 complete |
| Phase 3: Training | 1-2 weeks | InstanSeg raw weights from authors |
| Phase 4: Evaluation | 1 week | Pathologist availability |
| Phase 5: Integration | 2-3 days | Trained model |

Total: **4-6 weeks** from start to deployed model.

## Risks

1. **DAB threshold sensitivity** — Weak membrane staining may not produce clean boundaries. Mitigation: adaptive thresholding per tile based on DAB intensity distribution.
2. **Negative cells have no membrane signal** — DAB-negative cells have no visible membrane in IHC. Mitigation: fall back to nucleus expansion for negative cells, or use Hematoxylin channel gradients.
3. **InstanSeg raw weights** — The authors may take time to share weights. Mitigation: could train from scratch using the public training code, but would need more data and time.
4. **Generalization** — Model trained on CLDN18.2 may not generalize to other membrane IHC markers (HER2, PD-L1). Mitigation: include diverse staining patterns in training data; future fine-tuning on additional markers.
