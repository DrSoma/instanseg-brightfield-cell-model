# Model Quality Improvement Discovery
## Date: 2026-03-19 | 7 Research Agents | Deep Analysis

---

## Executive Summary

Seven parallel research agents investigated model architecture, training data, training strategy, post-processing, inference speed, InstanSeg internals, and 2025 literature. The consensus: **the biggest gains come from unlocking more training data and fixing the seed confidence problem, not from changing the architecture.**

---

## Tier 1: Implement Immediately (Next Training Run)

### 1. Enable Stain Augmentation (5 minutes)

**The code already exists but is disabled.** In InstanSeg's `augmentation_config.py`, the `brightfield_only` mode has:
```python
("normalize_HE_stains", [0.1, amount*0]),    # amount*0 = DISABLED
("extract_hematoxylin_stain", [0.1, amount*0])  # amount*0 = DISABLED
```

These use `torchstain`'s Macenko normalizer to randomly perturb stain vectors during training, simulating scanner-to-scanner variation. Critical for multi-scanner deployment (Hamamatsu + Aperio per the REB).

**Fix:** Monkey-patch in `04_train.py` to set `amount*0.5` before calling `instanseg_training()`. No new dependencies needed.

**Expected impact:** +5-10% robustness across staining variation.

---

### 2. Lazy Loading to Train on All 78k Tiles (30 lines of code)

**The #1 recommendation from the Training Data agent.** Currently limited to 10k tiles because InstanSeg's `_read_images_from_pth()` eagerly loads ALL images into RAM. But our dataset already stores string paths, not arrays.

**Fix:** Monkey-patch with a `LazyImageList` wrapper that defers `tifffile.imread` to `__getitem__` time:
```python
class LazyImageList:
    def __init__(self, items):
        self._items = items
    def __len__(self):
        return len(self._items)
    def __getitem__(self, i):
        item = self._items[i]
        return get_image(item) if isinstance(item, str) else item
```

RAM usage drops from ~30GB to ~4MB. Enables training on ALL 78,511 tiles (7.8x more data).

**Expected impact:** +15-25% quality improvement from sheer data volume.

---

### 3. Increase Seed Loss Weight (5 minutes)

**Root cause of the 33% skip rate.** Our model's seed logits peak at 1.6 (sigmoid=0.83) vs the stock model's 6.9 (sigmoid=0.999). The postprocessing maps this to `1.6/15 + 0.5 = 0.607` — barely above our threshold of 0.5.

**Fix:** In the InstanSeg loss function, increase `w_seed` from 1.0 to 3.0. This triples the gradient signal on the seed head, encouraging more confident predictions.

Additionally, extend hotstart from 10 to 20-30 epochs to give BCE pre-training more time to establish seed confidence before switching to Lovász hinge.

**Expected impact:** 30-50% fewer skipped tiles, more confident cell detections.

---

### 4. Cosine LR Annealing (5 minutes)

**Currently:** Fixed LR=0.0001 for all 200 epochs. Model early-stops at epoch 61.

**Fix:** Add `"-anneal", "true"` to `sys.argv` before calling `instanseg_training()`. InstanSeg already has cosine annealing implemented but not exposed through the training wrapper. Also increase epochs to 300 and patience to 50 (cosine needs room to cycle).

**Expected impact:** +1-3% from better convergence, potentially finds a better minimum.

---

### 5. Adaptive Per-Tile Seed Threshold (20 lines)

**Instead of a fixed `seed_threshold=0.5`**, compute it per-tile from the seed map distribution:
```python
fg_pixels = mask_map[mask_map > 0.1]  # exclude background
p90 = torch.quantile(fg_pixels, 0.90)
threshold = clamp(p90, min=0.3, max=0.7)
```

In sparse tiles, peaks may only reach 0.4 — a P90-based threshold adapts to ~0.35, recovering seeds. In dense tiles, P90 ≈ 0.65-0.70, preventing over-detection.

**Expected impact:** Recovers skipped tiles without the over-detection problem of globally lowering the threshold.

---

## Tier 2: Implement This Week

### 6. Retrain with BiomedParse/SegFormer Masks

The watershed masks are "loose" — boundaries don't tightly follow actual membrane staining. The BiomedParse SegFormer-B2 was trained to detect membrane pixels directly from DAB, producing tighter boundaries (gap narrowed from -0.035 to -0.010).

When the SegFormer finishes training on GPU 0, generate new cell masks with `generate_improved_masks()` and retrain InstanSeg on those.

**Expected impact:** +5-15% boundary accuracy.

### 7. Boundary-Weighted Loss

The Lovász hinge loss treats all pixels equally. For membrane-ring measurement, boundary pixels matter 10x more than interior pixels.

**Fix:** Add a weight map that upweights pixels within 5px of instance boundaries by 2-3x. The EDT is already computed in the loss function — use it to create the weight map.

**Expected impact:** +15-25% boundary quality (the metric that matters most for clinical measurement).

### 8. Nucleus-Guided Cell Fallback

For tiles where cell detection fails but nuclei are found, use marker-controlled watershed from nuclei seeds, constrained by DAB signal. The `segment_cells_enhanced()` function already exists in `watershed.py`.

**Expected impact:** Recovers most of the remaining 9% skip rate.

### 9. Disable Skeletonize in Mask Generation

Set `morphology_thinning: false` in config. The skeletonize step is extremely slow and actually counterproductive for watershed (wider membrane bands make better barriers).

**Expected impact:** 2-3x faster mask generation, slightly better watershed quality.

---

## Tier 3: Worth Trying Later

### 10. SegFormer Membrane as Extra Input Channel

Concatenate the SegFormer's membrane probability map with RGB (dim_in=4). The model would see both the raw image and a pre-computed membrane heatmap.

### 11. Deep Supervision

Add auxiliary losses at intermediate decoder levels. Forces the decoder to learn membrane features at every scale.

### 12. TensorRT for Inference Speed

Convert UNet backbone to TensorRT for 2-4x inference speedup. Keep postprocessing in PyTorch.

### 13. Cross-Attention Between Nuclei/Cell Heads

Add lightweight cross-attention so the cell head can see where nuclei are. Currently the two heads are completely independent.

---

## Literature Confirmation (2025 State-of-the-Art)

**Our approach is novel.** As of early 2025:
- No published self-supervised cell boundary learning from IHC membrane staining
- No automated CLDN18.2 scoring system exists
- No `brightfield_cells_nuclei` InstanSeg model exists
- CellViT++ uses IF for classification labels, not boundary learning (different contribution)
- Cellpose-SAM achieves "superhuman" generalization but is not IHC-membrane-specific
- The Aperio bar-filter (FDA-cleared for HER2) is our strongest validation approach

**Potential upgrades from literature:**
- CellVTA adapter (April 2025): Injects high-res CNN features into ViT encoders for cell segmentation SOTA
- Cellpose-SAM: Could serve as additional teacher model
- torchstain Macenko augmentation: Already in InstanSeg, just disabled

---

## Key Files for Implementation

| Component | File | What to Change |
|-----------|------|---------------|
| Stain augmentation | `04_train.py` | Monkey-patch augmentation dict: `amount*0` → `amount*0.5` |
| Lazy loading | `04_train.py` | Monkey-patch `_read_images_from_pth` with `LazyImageList` |
| Seed weight | `instanseg_loss.py` line 913 | `w_seed = 3.0` (currently 1.0) |
| Cosine LR | `04_train.py` line 237 | Add `"-anneal", "true"` to `sys.argv` |
| Adaptive threshold | inference wrapper | 20 lines: P90-based per-tile seed threshold |
| Boundary loss | `instanseg_loss.py` | Add EDT-based weight map at line 911 |
| Skeletonize | `config/default.yaml` line 34 | `morphology_thinning: false` |
| Hotstart | `config/default.yaml` line 73 | `hotstart_epochs: 25` |

---

## Summary: The Training Recipe for v2.0

```
1. Enable stain augmentation (amount=0.5)
2. Lazy loading (all 78k tiles, not 10k)
3. Seed weight = 3.0 (from 1.0)
4. Cosine LR annealing over 300 epochs
5. Hotstart 25 epochs (from 10)
6. Patience 50 (from 30)
7. Disable skeletonize in mask generation
8. Use BiomedParse/SegFormer masks when available
9. Adaptive per-tile seed threshold at inference

Expected training time: ~8-10 hours on A6000
Expected improvement: significant across all metrics
```
