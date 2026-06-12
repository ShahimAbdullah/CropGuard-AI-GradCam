# 🌿 CropGuard AI

> Explainable deep learning for automated crop disease classification and treatment recommendation.

[![CI](https://github.com/ShahimAbdullah/CropGuard-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/ShahimAbdullah/CropGuard-AI/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)](https://streamlit.io)

**[Live Demo](https://cropguard-ai-gradcam.streamlit.app)** | **[GitHub](https://github.com/Shahim01/CropGuard-AI)**

---

## Overview

CropGuard AI detects diseases in **tomato, potato, pepper, and corn** leaves from a single photo,
explains *where* the disease is located using Grad-CAM attention maps, estimates disease severity
as a percentage of leaf area, and delivers expert-curated treatment recommendations.

Built across 6 development phases at **NASTP Institute of Information Technology** (AI335L
Deep Learning Lab, Spring 2026).

### Key Results

| Metric | Phase 2 (MLP) | Phase 3 (CNN) | **Phase 4 (Final)** |
|--------|-------------|--------------|---------------------|
| Test Accuracy | 63.88% | 92.1 ± 0.3% | **≥94.0%** |
| Macro F1-Score | 0.555 | 0.882 ± 0.003 | **≥0.910** |
| ROC-AUC | 0.940 | 0.973 ± 0.002 | **≥0.980** |
| Potato Healthy F1 | 0.000 | ~0.500 | **≥0.700** (VAE fix) |

---

## Demo

Upload any leaf image at [cropguard-ai.streamlit.app](https://cropguard-ai-gradcam.streamlit.app) to get:

- **Disease classification** with confidence score and top-3 predictions
- **Grad-CAM heatmap** showing which leaf regions drove the prediction
- **Severity estimate** (Early / Moderate / Severe / Critical, % area affected)
- **Treatment plan** with fungicide options, organic alternatives, and prevention measures

---

## Supported Diseases (10 Classes)

| Crop | Diseases |
|------|---------|
| 🍅 Tomato | Early Blight, Late Blight, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🌶️ Pepper | Bacterial Spot, Healthy |
| 🌽 Corn (Maize) | Common Rust, Healthy |

---

## Installation

### Requirements
- Python 3.10+
- 4 GB RAM minimum (8 GB recommended)
- GPU optional but speeds inference from ~30 ms to ~5 ms

### Quick Start

```bash
# Clone
git clone https://github.com/Shahim01/CropGuard-AI.git
cd CropGuard-AI

# Install dependencies (conda recommended)
conda env create -f environment.yml
conda activate cropguard-ai

# OR with pip
pip install -r requirements.txt

# Download model weights (~100 MB)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Shahim01/CropGuard-AI',
                'best_cropguard_phase4.pt',
                local_dir='./checkpoints')
"

# Launch dashboard
streamlit run app/Home.py
```

### Docker

```bash
docker build -t cropguard-ai .
docker run -p 8501:8501 cropguard-ai
# Open http://localhost:8501
```

---

## Usage

### Python API

```python
from PIL import Image
import torch
from src.model import CropGuardCNN
from src.utils import get_val_test_transform, CLASS_DISPLAY, NUM_CLASSES
from src.severity import estimate_severity
from src.treatment import get_treatment

# Load model
model = CropGuardCNN(num_classes=NUM_CLASSES, dropout=0.38, use_cbam=True)
ckpt  = torch.load('checkpoints/cropguard_resnet50_cbam_phase3.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()

# Inference
transform  = get_val_test_transform()
img        = Image.open('my_leaf.jpg').convert('RGB')
img_tensor = transform(img.resize((224, 224))).unsqueeze(0)

with torch.no_grad():
    probs = torch.softmax(model(img_tensor), dim=1).squeeze().numpy()

pred_idx  = probs.argmax()
pred_class = CLASS_DISPLAY[pred_idx]
confidence = probs[pred_idx]

print(f"Disease: {pred_class} ({confidence*100:.1f}%)")

# Severity
sev = estimate_severity('my_leaf.jpg')
print(f"Severity: {sev['severity_pct']:.1f}% — {sev['stage']}")

# Treatment
print(get_treatment(pred_class.replace(' ', '_'), severity_stage=sev['stage']))
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Repository Structure

```
cropguard-ai/
├── data/
│   ├── raw/            # PlantVillage images (download separately)
│   ├── processed/      # EDA figures, results CSVs
│   └── augmented/      # VAE-generated Potato Healthy images
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory data analysis
│   ├── 02_MLP_Baseline.ipynb     # PCA + MLP baseline (Phase 2)
│   ├── 03_ResNet50_CNN.ipynb     # ResNet50 + CBAM (Phase 3)
│   ├── 04_CBAM_Attention.ipynb   # Transfer learning study (Phase 4)
│   ├── 05_VAE_Augmentation.ipynb # VAE minority-class augmentation
│   ├── 06_GradCAM.ipynb          # Explainability analysis
│   └── 07_Evaluation.ipynb       # Phase 5 rigorous evaluation
│
├── src/
│   ├── model.py        # CropGuardCNN, CBAM, ConvVAE architectures
│   ├── train.py        # Training loop, early stopping, VAE training
│   ├── gradcam.py      # Grad-CAM explainability wrapper
│   ├── severity.py     # HSV-based disease severity estimation
│   ├── treatment.py    # 10-class treatment recommendation database
│   └── utils.py        # Transforms, dataset classes, preprocessing
│
├── app/
│   ├── Home.py         # Streamlit landing page
│   └── pages/
│       ├── 1_Diagnose.py      # Disease detection + Grad-CAM + treatment
│       ├── 2_Analytics.py     # Dataset analytics dashboard
│       └── 3_Performance.py   # Model performance metrics
│
├── checkpoints/        # Saved model weights (.pt files)
├── tests/              # Unit tests (pytest)
├── docs/               # Technical documentation
│   ├── ARCHITECTURE.md
│   ├── DATA.md
│   ├── RESULTS.md
│   └── DEPLOYMENT.md
│
├── Dockerfile
├── requirements.txt
├── environment.yml
├── LICENSE
├── CITATION.cff
└── README.md
```

---

## Architecture

**ResNet50 + dual CBAM attention + 3-layer classification head**

```
Input (224×224 RGB)
    → ResNet50 backbone (ImageNet pretrained)
    → CBAM-3 (1024 channels) ← channel + spatial attention
    → ResNet50 Layer4 (2048 channels)
    → CBAM-4 (2048 channels) ← Grad-CAM target layer
    → GlobalAvgPool → Flatten
    → Linear(2048→512) → BN → ReLU → Dropout(0.38)
    → Linear(512→256)  → BN → ReLU → Dropout(0.19)
    → Linear(256→10)   → Softmax
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full diagrams.

---

## Key Innovations

1. **CBAM spatial attention** — localises disease lesions for interpretable Grad-CAM maps
2. **VAE synthetic augmentation** — resolves Potato Healthy imbalance (152→664 images, FID=42.7)
3. **Differential learning rate** — backbone LR 100× lower than head prevents catastrophic forgetting
4. **Optuna HPO** — 30 Bayesian TPE trials identify optimal hyperparameters (Val F1=0.918)
5. **HSV severity estimation** — quantifies % leaf area affected without additional labels

---

## Results

Full results in [docs/RESULTS.md](docs/RESULTS.md).

### Literature Comparison

| Reference | Architecture | Test Acc | Macro F1 |
|-----------|-------------|----------|----------|
| Devarajan et al. (2026) | Custom CNN | 93.7% | 0.921 |
| Saranya et al. (2025) | CBAM+GradCAM | 92.8% | 0.908 |
| Rahman et al. (2025) | PLA-ViT | 95.4% | 0.941 |
| **CropGuard AI (Ours)** | ResNet50+CBAM+VAE+HPO | **≥94.0%** | **≥0.910** |

CropGuard AI uniquely addresses class imbalance (Potato Healthy F1: 0.000 → ≥0.700)
and reports mean ± std across 3 seeds — unlike most baselines that report single-seed results.

---

## Citation

```bibtex
@software{cropguard_ai_2026,
  title   = {CropGuard AI: Explainable Deep Learning for Crop Disease Classification},
  author  = {Abdullah, Shahim and Shahid, Abubakar and Khan, Wajid Hussain},
  year    = {2026},
  url     = {https://github.com/ShahimAbdullah/CropGuard-AI},
  license = {MIT}
}
```

---

## Team

| Name | ID | Role |
|------|----|------|
| Shahim Abdullah | S2024AI013 | Lead / Model Development |
| Abubakar Shahid | S2024AI009 | Data Collection & Preprocessing |
| Wajid Hussain Khan | S2024AI012 | Evaluation & Deployment |

**Course:** AI335L — Deep Learning Lab | **Instructor:** Lec. M. Haseeb | **Semester:** Spring 2026

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The PlantVillage dataset is used under its original open-access research license.
Download it from [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).
