# CropGuard AI — Data Card

## Dataset Overview

| Property | Value |
|----------|-------|
| Name | PlantVillage (10-class subset) |
| Source | Kaggle: `abdallahalidev/plantvillage-dataset` |
| Variant | Colour (RGB) |
| Original Paper | Mohanty et al. (2016), *Frontiers in Plant Science* |
| License | Open access — research use permitted |
| Total Images | 11,481 (after quality filter: ~11,300) |
| Classes | 10 (4 crops × disease/healthy combinations) |
| Image Size | Variable (resized to 224×224 during preprocessing) |

## Class Distribution

| Class | Folder Name | Count | % Total | Notes |
|-------|------------|-------|---------|-------|
| Tomato Early Blight | `Tomato___Early_blight` | 1,000 | 8.71% | |
| Tomato Late Blight | `Tomato___Late_blight` | 1,909 | 16.63% | **Majority class** |
| Tomato Healthy | `Tomato___healthy` | 762 | 6.64% | |
| Potato Early Blight | `Potato___Early_blight` | 800 | 6.97% | |
| Potato Late Blight | `Potato___Late_blight` | 800 | 6.97% | |
| Potato Healthy | `Potato___healthy` | 152 | 1.32% | ⚠️ **Minority class** |
| Pepper Bacterial Spot | `Pepper,_bell___Bacterial_spot` | 1,702 | 14.83% | |
| Pepper Healthy | `Pepper,_bell___healthy` | 1,478 | 12.87% | |
| Corn Common Rust | `Corn_(maize)___Common_rust_` | 935 | 8.14% | |
| Corn Healthy | `Corn_(maize)___healthy` | 943 | 8.21% | |
| **Total** | | **11,481** | **100%** | |

**Imbalance ratio:** 12.6× (Tomato Late Blight vs Potato Healthy)

## Data Splits

| Split | Size | Fraction | Strategy |
|-------|------|----------|----------|
| Train | ~9,185 | 80% | Stratified, WeightedRandomSampler |
| Validation | ~1,148 | 10% | Stratified, no augmentation |
| Test | ~1,148 | 10% | Stratified, no augmentation, **evaluated once** |

- Split seed: `42` (reproducible)
- Stratification ensures each split has proportional class representation
- Test set was locked before Phase 5 evaluation (test-set ceremony protocol)

## Preprocessing Pipeline

1. **Class selection** — Filter to 10 target classes only
2. **Quality filter** — Laplacian variance ≥ 50.0 (removes blurry/corrupt images)
3. **Resize** — 224×224 (bilinear interpolation)
4. **RGB enforcement** — Convert all images to 3-channel RGB
5. **Normalisation** — ImageNet statistics: μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]

## Training Augmentation (applied to train split only)

| Step | Technique | Parameters |
|------|-----------|-----------|
| 1 | Resize + RandomCrop | 256→224 |
| 2 | RandomHorizontalFlip | p=0.5 |
| 3 | RandomVerticalFlip | p=0.3 |
| 4 | RandomRotation | ±30° |
| 5 | ColorJitter | brightness/contrast/saturation=0.3, hue=0.1 |
| 6 | RandomGrayscale | p=0.05 |
| 7 | GaussianBlur | kernel=3, σ∈[0.1, 2.0] |
| 8 | RandomErasing (CutOut) | p=0.20, scale 2–15% |
| 9 | ImageNet Normalisation | μ/σ as above |

Augmentation rationale: PlantVillage images are lab-controlled (uniform backgrounds,
consistent lighting). Augmentation simulates field conditions and prevents overfitting.

## VAE Synthetic Augmentation (Potato Healthy)

Due to the severe class imbalance (152 images), a Convolutional VAE was trained
on the Potato Healthy class to generate synthetic training samples.

| Property | Value |
|----------|-------|
| VAE latent dimension | 128 |
| Training images | 152 |
| Generated samples | 512 |
| FID Score | 42.7 (target < 50 ✓) |
| New training size | 664 (152 real + 512 synthetic) |
| New imbalance ratio | 2.9× (was 12.6×) |

## Ethical Considerations

- **Lab bias**: PlantVillage images are captured under controlled conditions.
  Real-field performance may be 10–20% lower due to occlusion, variable lighting,
  and multi-disease co-occurrence.
- **Crop coverage**: Only 4 of 30+ PlantVillage crops are included. Subsistence
  crops common in South/Southeast Asia and Africa are not represented.
- **Label quality**: ~30% of the hardest test examples may contain genuine label
  noise (borderline early-stage disease vs. healthy).
- **Deployment warning**: This system should not replace expert agronomic advice.
  It is a decision-support tool, not a diagnostic authority.

## Citation

```bibtex
@article{mohanty2016using,
  title   = {Using deep learning for image-based plant disease detection},
  author  = {Mohanty, Sharada P and Hughes, David P and Salath{\'e}, Marcel},
  journal = {Frontiers in Plant Science},
  volume  = {7},
  pages   = {1419},
  year    = {2016},
  doi     = {10.3389/fpls.2016.01419}
}
```
