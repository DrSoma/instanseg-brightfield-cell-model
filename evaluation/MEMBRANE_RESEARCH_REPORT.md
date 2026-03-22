# True Membrane Measurement: Research Report

**Date**: 2026-03-21
**Method**: 16-agent Double Diamond (11 research + 5 debate)
**Goal**: Make the InstanSeg model produce membrane-aligned cell boundaries

---

## The Problem

InstanSeg's cell boundaries are ~4px away from the actual DAB membrane. The fixed-ring membrane test FAILS (gap = -0.020). The bar-filter PASSES (gap = +0.127) but is classical image processing.

## What Was Investigated

11 research agents explored every viable approach:
1. Post-hoc boundary refinement (ray-casting to DAB peak)
2. Bar-filter as teacher (4-class semantic segmentation model)
3. CellViT, CellViT++, HoVerNet, StarDist, CellPose, CellPose-SAM, CellSAM
4. DAB-gradient loss functions (5 variants)
5. Foundation models (Virchow2, CONCH, TITAN, UNI2-h)
6. Refined training labels (DAB-guided re-watershed)
7. Active contours (classical snake, GVF, Chan-Vese, morphological GAC, conditional dilation)
8. CSGO architecture analysis
9. 13 SOTA models (2024-2026) including Classpose, PathoSAM, VISTA-PATH

## What Was Eliminated

| Approach | Reason |
|----------|--------|
| CellViT/CellViT++ | Nuclei-only, no membrane, 3-4x slower, no TorchScript |
| CellPose/CellPose-SAM | 10-25x slower, no TorchScript, boundary accuracy is label-dependent |
| HoVerNet | Slowest of all, nuclei-only |
| StarDist | Star-convex constraint, poor on IHC |
| Foundation models (pixel-level) | 16um token resolution vs 5um membrane |
| Active contours (per-cell) | 47-467 hours for 84M cells |
| DAB-gradient loss | InstanSeg loss never sees RGB image |
| Chan-Vese level set | Wrong signal model (2-phase) |

## Key Finding: No Model Produces Membrane-Aligned Boundaries

After exhaustive search of 13 SOTA models (2024-2026), **no cell segmentation model in existence produces membrane-aligned boundaries from brightfield IHC**. All use nucleus-centric expansion (watershed, flow fields, distance maps). CSGO is the only model with a dedicated membrane U-Net branch, but it requires manual annotation and was only tested on H&E.

## The Winning Strategy: Three Layers

### Layer 1: Refined Training Labels + Retrain InstanSeg
- DAB-guided re-watershed so boundaries sit at membrane
- Zero runtime cost (same architecture)
- Risk: 20% training success rate (sigma collapse)
- Expected gap: +0.05 to +0.10

### Layer 2: GPU Conditional Dilation at Inference
- Expand cell labels into DAB-positive regions via F.max_pool2d
- With Voronoi tie-breaking for adjacent cells
- 2-3 minutes for entire 191-slide cohort
- Expected gap: +0.08 to +0.12

### Layer 3: 4-Class Membrane U-Net
- Train 1.9M-param U-Net on bar-filter pseudo-labels
- Classes: background, nucleus, cytoplasm, membrane
- Replaces bar-filter; generalizes to different scanners
- Training: ~1 hour. Inference: ~2ms/tile

### Also evaluate: DenseCRF post-processing
- Standard medical image segmentation refinement
- Aligns boundaries with DAB gradients in milliseconds
- 20-line implementation with pydensecrf

## Execution Order

Layer 2 first (2 hours → go/no-go), then Layer 1 + Layer 3 in parallel on dual GPUs.

## Publication Framing

Cite CSGO as architectural precedent for separate nuclei + membrane detection. Claim two novel contributions:
1. DAB self-supervision eliminates manual annotation entirely
2. Bar-filter measurement at inference avoids boundary-rasterization error

## Agents Involved

11 research agents + 5 debate agents = 16 total Claude agents
