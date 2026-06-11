# CropGuard AI — Architecture

## System Overview

CropGuard AI is a deep-learning-based crop disease classification and treatment
recommendation system trained on the PlantVillage dataset (10-class subset).

```
Leaf Image (224×224 RGB)
        │
        ▼
┌───────────────────┐
│  Preprocessing    │  Resize → Normalize (ImageNet μ/σ)
└───────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  ResNet50 Backbone  (ImageNet-1K pretrained)           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Layer 0  │→│ Layer 1  │→│ Layer 2  │→│ Layer 3  │  │
│  │ (64 ch)  │ │(256 ch)  │ │(512 ch)  │ │(1024 ch) │  │
│  └──────────┘ └──────────┘ └──────────┘ └────┬─────┘  │
│                                               │         │
│                                          ┌────▼─────┐  │
│                                          │  CBAM-3  │  │
│                                          │ Ch + Sp  │  │
│                                          └────┬─────┘  │
│                                               │         │
│                                          ┌────▼─────┐  │
│                                          │ Layer 4  │  │
│                                          │(2048 ch) │  │
│                                          └────┬─────┘  │
│                                               │         │
│                                          ┌────▼─────┐  │
│                                          │  CBAM-4  │◄─── Grad-CAM target
│                                          │ Ch + Sp  │  │
│                                          └────┬─────┘  │
└───────────────────────────────────────────────┼────────┘
                                                │
                                     ┌──────────▼──────────┐
                                     │ AdaptiveAvgPool2d(1) │
                                     │   Flatten → (2048,) │
                                     └──────────┬──────────┘
                                                │
                                     ┌──────────▼──────────┐
                                     │  Classification Head │
                                     │  Linear(2048→512)    │
                                     │  BN → ReLU → Drop    │
                                     │  Linear(512→256)     │
                                     │  BN → ReLU → Drop    │
                                     │  Linear(256→10)      │
                                     └──────────┬──────────┘
                                                │
                                     ┌──────────▼──────────┐
                                     │  Softmax → 10-class  │
                                     │  Disease Prediction  │
                                     └─────────────────────┘
```

## CBAM Module

The Convolutional Block Attention Module (Woo et al., ECCV 2018) applies
**sequential channel then spatial recalibration**:

```
Input Feature Map (B, C, H, W)
        │
        ▼
┌─────────────────────┐
│  Channel Attention  │   "Which feature channels matter?"
│  AvgPool + MaxPool  │
│  → Shared MLP       │
│  → Sigmoid          │
│  → Scale channels   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Spatial Attention  │   "Where in the image is the disease?"
│  AvgPool + MaxPool  │   ← Grad-CAM target (CBAM-4)
│  Concat → Conv(7×7) │
│  → Sigmoid          │
│  → Scale spatially  │
└──────────┬──────────┘
           │
           ▼
  Recalibrated Feature Map (B, C, H, W)   [same shape]
```

## VAE Architecture (Minority Class Augmentation)

```
Real Potato Healthy Image (3, 64, 64) ─────► Encoder
                                               │
                           Conv×4 + BN + LeakyReLU → Flatten → μ, log_var
                                               │
                              Reparameterization: z = μ + ε·σ, ε~N(0,I)
                                               │
                                            Decoder
                                               │
                           Linear → Reshape → ConvTranspose×4 + BN + ReLU → Tanh
                                               │
                                        Synthetic Image (3, 64, 64)

Loss: L_ELBO = L_recon (MSE) + β·L_KL
β annealing: 0 → 1 over 50 epochs  (prevents posterior collapse)
```

## Transfer Learning Strategy

| Phase | Epochs | Frozen Layers | Head LR | Backbone LR |
|-------|--------|---------------|---------|-------------|
| 1 — Feature Extraction | 10 | All ResNet50 | 1e-3 | 0 |
| 2 — Fine-Tuning | 20 | Bottom 120 params | 7.3e-4 | 4.2e-6 |

**Config C (Differential LR)** was selected after comparing 4 freezing strategies.
The 100× LR ratio (head vs backbone) preserves ImageNet Gabor-like edge detectors
while adapting high-level features to plant disease patterns.

## Hyperparameters (HPO Trial 23 — Optuna Bayesian TPE, 30 trials)

| Hyperparameter | Value | Importance |
|----------------|-------|-----------|
| Head LR | 7.3e-4 | 0.412 (1st) |
| Dropout | 0.38 | 0.287 (2nd) |
| Backbone LR | 4.2e-6 | 0.143 (3rd) |
| Weight Decay | 3.1e-5 | 0.089 (4th) |
| Label Smoothing | 0.08 | 0.047 (5th) |
| Batch Size | 32 | 0.022 (6th) |

## Streamlit Dashboard Architecture

```
┌─────────────────────────────────────────────────┐
│                 Streamlit App                    │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Home.py │  │ Diagnose │  │  Analytics    │  │
│  │ Overview │  │ + GradCAM│  │  + EDA charts │  │
│  └──────────┘  └────┬─────┘  └───────────────┘  │
│                     │                             │
│           ┌─────────▼─────────┐                  │
│           │  CropGuardCNN     │  (cached @start)  │
│           │  ResNet50 + CBAM  │                  │
│           └─────────┬─────────┘                  │
│                     │                             │
│           ┌─────────▼─────────┐                  │
│           │  GradCAMWrapper   │                  │
│           │  + SeverityModule │                  │
│           │  + TreatmentDB    │                  │
│           └───────────────────┘                  │
└─────────────────────────────────────────────────┘
```
