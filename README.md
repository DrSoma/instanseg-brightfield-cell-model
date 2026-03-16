# InstanSeg Brightfield Cell+Nucleus Model

**Training a brightfield nuclei-and-cells InstanSeg model using DAB membrane staining as self-supervised ground truth.**

## The Problem

InstanSeg — the state-of-the-art cell/nucleus segmentation model used in QuPath and Orion — currently offers three pre-trained models:

| Model | Input | Output |
|-------|-------|--------|
| `brightfield_nuclei` | H&E, H-DAB | Nuclei only |
| `fluorescence_nuclei_and_cells` | Fluorescence | Nuclei AND cells |
| `single_channel_nuclei` | Single fluorescence channel | Nuclei only |

**There is no brightfield model that segments both nuclei AND cell boundaries.** This is a known gap in the field. The reason: cell membranes are generally invisible in standard brightfield H&E staining, so nobody has large-scale cell boundary annotations for brightfield images.

Without cell boundaries, downstream analysis cannot compute:
- **Cytoplasm measurements** (cell minus nucleus)
- **Membrane measurements** (cell boundary ring)
- **Nucleus/Cell area ratio**
- **Per-compartment stain intensity** (e.g., "Cytoplasm: DAB mean")

These are clinically important for IHC scoring, particularly for membrane-targeted biomarkers.

## Our Advantage: DAB Membrane Staining

We have access to **200+ whole-slide images** of CLDN18.2 (Claudin 18.2) immunohistochemistry. CLDN18.2 is a **membrane protein** — the DAB chromogen in these slides specifically stains the cell membrane, making it clearly visible as brown outlines around cells.

This means we have **natural membrane annotations built into the staining itself**. The DAB signal can be computationally extracted via stain deconvolution and used as ground truth for cell boundary training — no manual annotation required.

## The Approach: Self-Supervised Cell Boundary Generation

```
Slide Tile (RGB)
      │
      ├── Stain Deconvolve ──► DAB Channel (membrane signal)
      │                              │
      │                              ├── Threshold ──► Binary Membrane Mask
      │                              │
      ├── InstanSeg ──► Nucleus Instance Masks (seeds)
      │                              │
      │                              ▼
      └────────────────► Watershed (nuclei seeds + membrane mask)
                                     │
                                     ▼
                         Paired Nucleus + Cell Instance Masks
                                     │
                                     ▼
                         Fine-tune InstanSeg (dual-output architecture)
                                     │
                                     ▼
                         brightfield_nuclei_and_cells model
```

### Pipeline Steps

1. **Stain deconvolution** — Separate RGB tiles into Hematoxylin, DAB, and Residual channels using Beer-Lambert law with standard H-DAB stain vectors
2. **DAB thresholding** — Binarize the DAB channel to extract membrane outlines
3. **Nucleus detection** — Run existing `brightfield_nuclei` InstanSeg to get nucleus instance masks
4. **Watershed cell segmentation** — Use nuclei as seeds, DAB membrane mask as boundaries, grow cells outward until they meet the membrane or a neighboring cell
5. **Quality filtering** — Remove cells where the membrane signal is absent or ambiguous
6. **Training data packaging** — Convert paired nucleus/cell masks into InstanSeg's `segmentation_dataset.pth` format
7. **Fine-tune InstanSeg** — Use the dual-output architecture from `fluorescence_nuclei_and_cells` with brightfield-specific weights
8. **Evaluate** — Compare against QuPath's nucleus expansion baseline and the fluorescence model
9. **Export** — TorchScript model compatible with both QuPath and Orion

## Slide Data

- **Location**: `/pathodata/Claudin18_project/CLDN18_SLIDES_ANON/` (CLDN0353–CLDN0452)
- **Format**: Hamamatsu NDPI (pyramidal TIFF)
- **Stain**: CLDN18.2 IHC (Hematoxylin counterstain + DAB chromogen targeting membrane protein)
- **Scanner**: Hamamatsu NanoZoomer, ~0.23 µm/px at 40x
- **Count**: 100 anonymized slides available for training, 114 additional in the full cohort

## Technical Details

### Stain Deconvolution

Standard H-DAB vectors (matching QuPath defaults and Orion's Rust implementation):

```
Hematoxylin: [0.650, 0.704, 0.286]
DAB:         [0.268, 0.570, 0.776]
Residual:    [0.711, 0.424, 0.562]
```

Calibrated vectors from the CLDN18 pipeline (scanner-specific):

```
Hematoxylin: [0.786, 0.593, 0.174]
DAB:         [0.215, 0.422, 0.881]
Residual:    [0.547, -0.799, 0.249]
```

### InstanSeg Training Infrastructure

- Training code: https://github.com/instanseg/instanseg (`train.py`)
- Data format: `segmentation_dataset.pth` (PyTorch tensor file)
- Dataset preparation: `instanseg/notebooks/load_datasets.ipynb`
- Model export: `export_model.ipynb` (to TorchScript for QuPath/Orion)
- Pre-trained weights: Available from InstanSeg authors upon request (TorchScript models cannot be fine-tuned directly; raw weights needed)

### Expected Output

The trained model produces **two segmentation masks** per tile:
- **Nucleus mask** — instance labels for each nucleus
- **Cell mask** — instance labels for each whole cell (including cytoplasm)

QuPath/Orion then derives compartments:
- **Nucleus** = nucleus mask ROI
- **Cell** = cell mask ROI
- **Cytoplasm** = Cell ROI minus Nucleus ROI
- **Membrane** = thin ring at cell boundary
- **Nucleus/Cell ratio** = nucleus area / cell area

## Why This Matters

1. **Clinical need** — Membrane-targeted biomarkers (CLDN18.2, HER2, PD-L1) require membrane-level quantification for accurate IHC scoring
2. **No existing solution** — Nobody has a brightfield cell+nucleus InstanSeg model
3. **Self-supervised** — The DAB stain IS the annotation, eliminating weeks of manual pathologist annotation
4. **Scalable** — 200+ slides provide massive training data
5. **Community impact** — The model would benefit the entire digital pathology community using QuPath and InstanSeg

## References

- **InstanSeg** — Goldsborough et al. (2024). "InstanSeg: an embedding-based instance segmentation algorithm." arXiv:2408.15954
- **TissueNet** — Greenwald et al. (2022). "Whole-cell segmentation of tissue images with human-in-the-loop." Nature Biotechnology, 40(4). (Fluorescence only — not applicable to brightfield)
- **QuPath** — Bankhead et al. (2017). "QuPath: Open source software for digital pathology." Scientific Reports, 7(1).
- **Beer-Lambert stain deconvolution** — Ruifrok & Johnston (2001). "Quantification of histochemical staining by color deconvolution." Analytical and Quantitative Cytology and Histology.

## Related Projects

- **Orion** — WSI viewer with InstanSeg integration (https://github.com/DrSoma/orion) — the trained model will be deployed here
- **CLDN18 Pipeline** — IHC quality control pipeline that uses InstanSeg for cell detection

## License

MIT
